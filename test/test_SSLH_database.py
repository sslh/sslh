"""
Test class for 'SSLH_postgres' (database connection for storing experimental results for SSL-H)

First version: June 12, 2015
This version: June 14, 2015
Author: Wolfgang Gatterbauer <gatt@cmu.edu>
"""


from SSLH_database import database_sql_statement, database_insert_tuple
from SSLH_files import save_csv_records
import datetime
CONNECTION = "host='localhost' dbname='LinBP' user='postgres' password=''"


# -- Determine path to data irrespective (!) of where the file is run from
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
current_path = dirname(abspath(getfile(currentframe())))
data_directory = join(current_path, 'data/')


def test_database_sql_statement_create_table():
    print "\n-- 'database_sql_statement' (create a temp table) --"
    sql_file_name = 'create_temp_table.sql'        # !!! adapt schema
    with open (join(data_directory, sql_file_name), "r") as myfile:           # read sql_nips string from file
        sqlString=myfile.read()
    print sqlString
    database_sql_statement(CONNECTION, sqlString)


def test_database_insert_tuple():
    print "\n-- 'database_insert_tuple' (insert tuple) --"
    TABLE = "temp"
    tuple = [0.33333,0.33333,0.1,0.7,0.2,0.1,0.7,0.2,1000,3,1,'CBM',
             0,0.995,0,'undirected',0.01,1,5,10,'','',0.0427341461182,0.00451302528381,0.217085427136]

    # -- adds datetime as first column
    tuple2 = [str(datetime.datetime.now())]
    tuple2.extend(tuple)
    print "tuple:\n", tuple2

    database_insert_tuple(CONNECTION, TABLE, tuple2)


def test_database_sql_statement_save_csv_records():
    print "\n-- 'database_sql_statement', 'save_csv_records' --"
    tableName = "temp"
    # tableName = "results_propagation"
    sqlString = "SELECT * FROM %s" %tableName
    records = database_sql_statement(CONNECTION, sqlString)

    # -- write to external CSV file
    file_name = 'test_database_sql_statement_save_csv_records.csv'
    save_csv_records(join(data_directory, file_name), records)
    print "Query result saved in:\n  ", file_name


if __name__ == '__main__':
    test_database_sql_statement_create_table()
    test_database_insert_tuple()
    test_database_sql_statement_save_csv_records()
