from flask import jsonify
import requests

import pycountry
import xml.etree.ElementTree as ET
import pandas as pd




def get_json_Annual_GDP(country_code):
    '''
    Assumption: the country code is correct
    '''
    
    country = pycountry.countries.get(alpha_2=country_code)
    country_name=country.name
    url="http://api.worldbank.org/v2/country/"+str(country_code)+"/indicator/NY.GDP.MKTP.CD"


    r=requests.get(url)
    str_text=r.text
    # find the location of first '<'
    start=str_text.find('<')
    str_text=str_text[3:]
    root = ET.fromstring(str_text)

    print("Reporting ORG",root.items()[5][1])
    print("Reporting ORG ID",root.items()[4][1])
    print("Reporting Date",root.items()[6][1])
    


    dict_result={}
    dict_result["reportingOrg"]=root.items()[5][1]
    dict_result["reportingOrgID"]=root.items()[4][1]
    dict_result["reportingDate"]=root.items()[6][1]
    dict_result["countryName"]=country_name
    dict_result["countryCode"]=country_code
    dict_result["data"]={}
    for i in range(0,len(root)):
        child=root[i]
        year=child[3].text
        annual_gdp=child[4].text
#         print(year,annual_gdp)
        dict_result["data"][year]=annual_gdp



    return jsonify(dict_result)



    

def get_csv_Annual_GDP(country_code):
    '''
    Assumption: the country code is correct
    returns a dataframe
    '''
    
    country = pycountry.countries.get(alpha_2=country_code)
    country_name=country.name
    url="http://api.worldbank.org/v2/country/"+str(country_code)+"/indicator/NY.GDP.MKTP.CD"


    r=requests.get(url)
    str_text=r.text
    # find the location of first '<'
    start=str_text.find('<')
    str_text=str_text[3:]
    root = ET.fromstring(str_text)

    print("Reporting ORG",root.items()[5][1])
    print("Reporting ORG ID",root.items()[4][1])
    print("Reporting Date",root.items()[6][1])
    reporting_date=root.items()[6][1]
    reporting_org_ID=root.items()[4][1]
    reporting_org=root.items()[5][1]
    
    
    


    dict_result={}
    
    
    dict_result["reportingOrg"]=[]
    dict_result["reportingOrgID"]=[]
    dict_result["reportingDate"]=[]
    dict_result["countryName"]=[]
    dict_result["countryCode"]=[]
    dict_result["year"]=[]
    dict_result["annualGDP"]=[]

    for i in range(0,len(root)):
        child=root[i]
        year=child[3].text
        annual_gdp=child[4].text
#         print(year,annual_gdp)
        dict_result["year"].append(year)
        dict_result["annualGDP"].append(annual_gdp)
        
        dict_result["reportingOrg"].append(reporting_org)
        dict_result["reportingOrgID"].append(reporting_org_ID)
        dict_result["reportingDate"].append(reporting_date)
        dict_result["countryName"].append(country_name)
        dict_result["countryCode"].append(country_code)

    df=pd.DataFrame(data=dict_result)

    return df



