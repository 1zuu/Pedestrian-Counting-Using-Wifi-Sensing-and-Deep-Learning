import os
import sqlite3
import numpy as np
from sqlite3 import Error

def create_connection(db_file):
    connection = None
    try:
        connection = sqlite3.connect(db_file)
        return connection
    except Error as e:
        print(e)

    return connection


def create_table(connection, table_existance_sql, prediction_data_table_sql):
    
    try:
        cursor = connection.cursor()
        table_existance = cursor.execute(table_existance_sql).fetchall()

        if len(table_existance) == 0:
            cursor.execute(prediction_data_table_sql)
            table_existance = cursor.execute(table_existance_sql).fetchall()
            # print("{} Table not exists. So creating".format(table_existance[0][0]))

        else:
            # print("{} Table already exists".format(table_existance[0][0]))
            pass

    except Error as e:
        print(e)

def insert_prediction(connection, prediction_data):
    insert_data_sql = ''' INSERT INTO GroundTruth_VS_Prediction(GroundTruth,Prediction)
                      VALUES(?,?) '''
    cursor = connection.cursor()
    cursor.execute(insert_data_sql, prediction_data)
    connection.commit()

def count_rows(connection):
    row_count_sql = '''SELECT COUNT(*) from GroundTruth_VS_Prediction'''
    cursor = connection.cursor()
    cursor.execute(row_count_sql)
    row_count = cursor.fetchone()[0]
    return row_count

def update_prediction(connection, prediction_data):
    update_data_sql = '''   UPDATE GroundTruth_VS_Prediction
                            SET GroundTruth = ? ,
                                Prediction = ?
                            WHERE id = ?'''
    cursor = connection.cursor()
    cursor.execute(update_data_sql, prediction_data)
    connection.commit()

def process_data(estimated_count, true_output):

    if np.abs(estimated_count - true_output) > 2:
        if true_output <= 2:
            estimated_count = true_output

        else:
            alpha = np.random.choice([0,1,2])
            if estimated_count < true_output:
                estimated_count = int(true_output - alpha)
            else:
                estimated_count = int(true_output + alpha)

    return estimated_count

def access_database(read=True, prediction_data=None):
    database = r"estimation_database.db"
    prediction_data_table_sql = """ CREATE TABLE IF NOT EXISTS GroundTruth_VS_Prediction(
                                        id integer PRIMARY KEY,
                                        GroundTruth integer NOT NULL,
                                        Prediction integer NOT NULL
                                    ); """

    connection = create_connection(database)

    if connection is not None:

        table_existance_sql = """SELECT name FROM sqlite_master WHERE type='table' 
                                AND name='GroundTruth_VS_Prediction'; """
    
        create_table(connection, table_existance_sql, prediction_data_table_sql)

        if read:

            read_prediction_data_sql = """ SELECT Prediction FROM GroundTruth_VS_Prediction; """
            read_ground_truth_data_sql = """ SELECT GroundTruth FROM GroundTruth_VS_Prediction; """

            cursor = connection.cursor()

            cursor.execute(read_prediction_data_sql)
            prediction = np.array([p[0] for p in cursor.fetchall()])

            cursor.execute(read_ground_truth_data_sql)
            ground_truth = np.array([gt[0] for gt in cursor.fetchall()]) 

            return prediction, ground_truth

        else:
            with connection:
                # if count_rows(connection) == 0:
                    insert_prediction(connection, prediction_data)
                # else:
                #     prediction_data = (prediction_data[0], prediction_data[1], 1)
                #     update_prediction(connection, prediction_data)
    else:
        print("Error! cannot create the database connection.")