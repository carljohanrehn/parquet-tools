"""
Examples for reading and writing Apache Parquet files using the Spark API.

References:
    https://spark.apache.org/docs/2.2.0/sql-programming-guide.html
    https://spark.apache.org/docs/2.2.0/api/python/index.html
    https://parquet.apache.org/
    https://gist.github.com/beauzeaux/a68d6f32f4985ed547ce

Note: If you use IntelliJ IDEA, remember to set environment the variable PYSPARK_PYTHON in the run configuration to
      the path of your python executable.
"""

import os

from sqlalchemy import create_engine, inspect

import numpy as np
import pandas as pd

import pyspark
import pyspark.sql


# Database utilities
def create_sqlite(database_path):
    return create_engine('sqlite:///' + database_path)


def get_table_names(engine):
    return engine.get_table_names()


def get_column_names(engine, table_name):
    inspector = inspect(engine)
    return [column['name'] for column in inspector.get_columns(table_name)]


def get_table(engine, table_name):
    query = """SELECT * FROM {0}""".format(table_name)
    for row in pd.read_sql(query, engine).itertuples(index=False, name=table_name):
        yield list(row._asdict().values())


def save_table(df, database, table_name, f_engine, index=True):
    df.to_sql(table_name, f_engine(database), if_exists='replace', index=index)


# Using the Spark Python API for reading and writing Parquet files
def create_dataframe(spark_context, sql_context, table, column_names):

    data = spark_context.parallelize(table, 20)
    data.persist(pyspark.StorageLevel(True, True, False, True, 1))

    df = sql_context.createDataFrame(
        data, schema=column_names, samplingRatio=None
    ).repartition(20)
    # df.persist(pyspark.StorageLevel(True, True, False, True, 1))

    return df


def save_to_parquet(spark_context, sql_context, engine, tables, file_path):
    for table in tables:
        create_dataframe(
            spark_context, sql_context,
            get_table(engine, table),
            get_column_names(engine, table)
        ).write.parquet(os.path.join(file_path, table))


def main():

    # Path to Parquet files
    parquet_dir = './data'

    # Create test data
    table_list = ['table1', 'table2']
    column_list = [['a', 'b', 'c'], ['d', 'e', 'f']]

    for table, columns in zip(table_list, column_list):
        save_table(
            pd.DataFrame(np.random.randint(low=0, high=10, size=(5, 3)), columns=columns),
            os.path.join(parquet_dir, 'db.sqlite'),
            table,
            lambda name: create_engine('sqlite:///' + name)
        )

    # Set up Spark context
    conf = pyspark.SparkConf()

    args = (
        ('spark.executor.memory', '4g'),
        ('spark.sql.parquet.compression.codec', 'gzip'),
        # ('spark.sql.parquet.compression.codec', 'snappy')
    )

    map(lambda args: conf.set(*args), args)

    spark_context = pyspark.SparkContext("local", conf=conf)
    sql_context = pyspark.SQLContext(spark_context)

    # Save Parquet file
    save_to_parquet(
        spark_context, sql_context,
        create_sqlite(os.path.join(parquet_dir, 'db.sqlite')),
        table_list, parquet_dir
    )

    # Read Parquet file
    table_df = sql_context.read.parquet(os.path.join(parquet_dir, 'table1'))
    print(table_df.count())
    print(table_df.head())


if __name__ == "__main__":
    main()
