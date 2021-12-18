# write to sheet
import os
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import pandas as pd

def create_keyfile_dict():
	client_secret_dic = {
	"type" : os.environ.get("type"),
	"project_id" : os.environ.get("project_id"),
	"private_key_id" : os.environ.get("private_key_id"),
	"private_key" : os.environ.get("private_key").replace("\\n","\n"),
	"client_email" : os.environ.get("client_email"),
	"client_id" : os.environ.get("client_id"),
	"auth_uri" : os.environ.get("auth_uri"),
	"token_uri" : os.environ.get("token_uri"),
	"auth_provider_x509_cert_url" : os.environ.get("auth_provider_x509_cert_url"),
	"client_x509_cert_url" : os.environ.get("client_x509_cert_url")
	}
	return client_secret_dic


def get_sheet(sheetname):
	#remember to share sheet with email id in client secret
    print("Looking for",sheetname)
    # use creds to create a client to interact with the Google Drive API
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    dict_cred=create_keyfile_dict()
    # print("dict_cred is ",dict_cred)
    creds = ServiceAccountCredentials. from_json_keyfile_dict(dict_cred, scope)
    print(creds)
    print(type(creds))
    # print(creds)
    client = gspread.authorize(creds)
    # Find a workbook by name and open the first sheet
    # Make sure you use the right name here.
    
    status=True
    sheet=None
    try:
        sheet = client.open(sheetname).sheet1
        return status,sheet
    except Exception as e: 
        print("Error is ",e)
        print("An error")
        print(sheet)
        print("An exception occurred") 
        status=False
        return status,sheet


def update_sheet(sheetname,df):
    print("Inserting dataframe into ",sheetname)
    print(df)
    status,sheet=get_sheet(sheetname)
    dataframe = pd.DataFrame(sheet.get_all_records())
    dataframe=dataframe.fillna("")
    print("Current data = ",dataframe.shape,"\n",dataframe.head())
    dataframe = pd.concat([dataframe, df], ignore_index=True, sort=False)
    print("FInal dataframe = \n",dataframe)
    sheet.update([dataframe.columns.values.tolist()] + dataframe.values.tolist())
    return True


def insert_row(row,sheetname):
    # row is a list of strings
    print("Inserting",row)
    status,sheet=get_sheet(sheetname)
    if not status:
        print("Issue")
        return False
    index = len(sheet.get_all_values())
    print("Index is ",index,"inserting",row)
    # sheet.insert_row([row], index+1)
    sheet.insert_row([row[0],""], index+1)

    return True
