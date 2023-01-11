import os
import boto3
import sys
from io import StringIO, BytesIO
from time import sleep
import re
import pandas as pd
from hashlib import md5
import json

# ------------------------------------------------------------

def write_json(target_path, target_file, data):
    # Write JSON to specific path
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
        except Exception as e:
            print(e)
            raise
    with open(os.path.join(target_path, target_file), 'w') as f:
        json.dump(data, f, indent=4)

# ------------------------------------------------------------
# Load secrets file containing information
with open("credentials.json") as f:
    credentials = json.load(f)

s3 = boto3.resource(
    service_name=credentials["SERVICE_NAME"],
    region_name=credentials["REGION_NAME"],
    aws_access_key_id=credentials["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=credentials["AWS_SECRET_ACCESS_KEY"]
)
# ------------------------------------------------------------


def cache_name(*args):
    cn = "".join(args)
    cn = md5(cn.encode("utf-8")).hexdigest()
    return "_khoj_cache/" + cn + ".parquet"


def check_cache(*args):
    return os.path.exists(cache_name(*args))


def read_cache(*args):
    return pd.read_parquet(cache_name(*args))


def write_cache(df, *args):
    if not os.path.exists("_khoj_cache"):
        os.mkdir("_khoj_cache")
    return df.to_parquet(cache_name(*args))


def from_athena_query(query, do_read_cache=True, do_write_cache=True, query_check_sleep=5, workgroup='primary',
                      output_location=None):
    """
    Read data from Athena with a given query.

    :param query: string, valid
    :param do_read_cache: boolean, read from local cache if possible
    :param do_write_cache: boolean, write to local cache
    :param query_check_sleep: integer, optional parameter, number of seconds to wait before checking again if query
        result is ready
    :param query_check_sleep: integer, optional parameter, number of seconds to wait before checking again
        if query result is ready
    :param workgroup: string, optional parameter, athena workgroup name e.g. "primary"
    :param output_location: string, optional parameter, "s3://<bucket-name>/<some folder>" s3 bucket / folder
        where query ouputs are stored
    :return: pandas data frame

    TODO:
    * check after some time if boto3 offers a waiter for athena queries to avoid using a loop
    * if possible extend functionality for loading struct fields (arrive as string)
    """

    if check_cache(query) and do_read_cache:
        return read_cache(query)

    athena_client = boto3.client('athena')
    s3_client = boto3.client('s3')

    query_param = {}
    query_param['QueryString'] = query
    if workgroup:
        query_param['WorkGroup'] = workgroup
    if output_location:
        query_param['ResultConfiguration'] = {'OutputLocation': output_location}

    respond_start_query_execution = athena_client.start_query_execution(
        **query_param
    )
    query_execution_id = respond_start_query_execution["QueryExecutionId"]

    execution_state = 'QUEUED'

    # if execution_state:  # don't start if empty ... just in case something went wrong
    while execution_state in ('QUEUED', 'RUNNING'):
        response_execution = athena_client.get_query_execution(
            QueryExecutionId=query_execution_id
        )
        execution_state = response_execution["QueryExecution"]["Status"]["State"]

        sleep(query_check_sleep)  # wait until next check

    # basic error handling - show athena error message:
    if execution_state in ("FAILED", "CANCELLED"):
        state_change_reason = response_execution["QueryExecution"]["Status"].get("StateChangeReason")
        message_exception = "Query execution ended with state %s! StateChangeReason: \%s" % (
            execution_state, str(state_change_reason))
        raise Exception(message_exception)

    s3_path = response_execution['QueryExecution']['ResultConfiguration']['OutputLocation']
    s3_bucket = re.sub(u"s3://([^/]+)/.*", r"\1", s3_path)
    s3_key = re.sub(u"s3://[^/]+/(.*)", r"\1", s3_path)

    s3_obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
    df = pd.read_csv(BytesIO(s3_obj['Body'].read()))

    if do_write_cache:
        write_cache(df, query)

    return df


def parse_sql(sql_file, replacement=None):
    """
    sql_file: string; relative location of .sql file e.g. "../my_query.sql"
    replacement: dict; mapping for f-string like replacement. Use curly braces within sql query to indicate variable (e.g. {date_dummy})
    """

    with open(sql_file, mode="r") as file:
        query = file.read()
    file.close()

    if replacement:
        for dummy, repl in replacement.items():
            query = query.replace(dummy, repl)

    return query