"""
Postgres database interaction for experimental results (SSL-H)

First version: June 13, 2015
This version: June 13, 2015
Author: Wolfgang Gatterbauer <gatt@cmu.edu>
"""


import psycopg2


def database_sql_statement(connection_string, sql_string):
    """Given a connection string and a SQL statement (can be an insert, or create table statement, or query):
    Issues the SQL statement and optionally returns the result as list of tuples (in case it was a query)
    """
    connection = psycopg2.connect(connection_string)
    cursor = connection.cursor()
    cursor.execute(sql_string)
    connection.commit()                 # only needed in case this is an insert or create table statement
    try:
        return cursor.fetchall()        # returns a list of tuples
    except Exception, e:
        return # if there is a programming Error because there is nothing to fetch (e.g., insert statement), then just return


def database_insert_tuple(connection_string, table_name, tuple):
    """Given a connection string, a table, and a tuple:
    Inserts the tuple into the table.
    replaces empty string '' with NULL
    """
    tuple2 = "(" + str(tuple)[1:-1] + ")"       # create a single string
    tuple2 = tuple2.replace("''", "NULL")       # replace '' with NULL
    sqlString = "insert into " + table_name + " values " + tuple2 + ";"
    connection = psycopg2.connect(connection_string)
    cursor = connection.cursor()
    cursor.execute(sqlString)
    connection.commit()
