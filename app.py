from flask import Flask, render_template, request,send_from_directory, jsonify, url_for
from flask import redirect, make_response

from flask_cors import CORS

import numpy as np
import pycountry

import pandas as pd
import os
import cv2

from utils import helper
# from utils import mpt_helper
from utils import ppl_counter_helper
# from utils import gtrends_helper
from utils import retinopathy_helper


import json
import plotly.express as px
from flask import Markup

from datetime import datetime




def generate_country_codes():
    list_of_country_codes_alpha2=[]
    for country in list(pycountry.countries):
#     print(country.alpha_2)
        list_of_country_codes_alpha2.append(country.alpha_2)
    return list_of_country_codes_alpha2



list_of_country_codes_alpha2=generate_country_codes()
# will contain the list of country codes 2 alphabets


application = Flask(__name__)
CORS(application)


@application.route("/get/mptposts/timeseries/<country_code>")
def get_mptposts_timeseries(country_code):
    '''
    get a time series of ads made
    every day
    fromt he beginning of time
    '''
    download_type="json"
    possible_download_types=["csv","json"]
    if "download-type" in request.headers:
        print("User requested download type ",request.headers["download-type"])
        download_type=request.headers["download-type"]
        if download_type not in possible_download_types:
            return "Download type incorrect. Please check header"

    if download_type=="json":
        mptposts_timeseries_json=mpt_helper.get_json_mptposts_timeseries(country_code)
        return mptposts_timeseries_json
    elif download_type=="csv":
        mptposts_timeseries_df=mpt_helper.get_csv_mptposts_timeseries(country_code)
        resp = make_response(mptposts_timeseries_df.to_csv(index=False))
        resp.headers["Content-Disposition"] = "attachment; filename="+country_code+"mptposts_timeseries.csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp


@application.route("/get/mptposts/segmentcount/<country_code>")
def get_mptposts_segmentcount(country_code):
    '''
    get count of advertisements made in each segment
    '''
    download_type="json"
    possible_download_types=["csv","json"]
    if "download-type" in request.headers:
        print("User requested download type ",request.headers["download-type"])
        download_type=request.headers["download-type"]
        if download_type not in possible_download_types:
            return "Download type incorrect. Please check header"

    if download_type=="json":
        mptposts_segmentcount_json=mpt_helper.get_json_segmentcount(country_code)
        return mptposts_segmentcount_json
    elif download_type=="csv":
        mptposts_segmentcount_df=mpt_helper.get_csv_segmentcount(country_code)
        resp = make_response(mptposts_segmentcount_df.to_csv(index=False))
        resp.headers["Content-Disposition"] = "attachment; filename="+country_code+"mptposts_timeseries.csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp


@application.route("/get/mptposts/subjperseg/<country_code>")
def get_mptposts_subjperseg(country_code):
    '''
    get count of advertisements made in each segment
    '''
    download_type="json"
    possible_download_types=["csv","json"]
    if "download-type" in request.headers:
        print("User requested download type ",request.headers["download-type"])
        download_type=request.headers["download-type"]
        if download_type not in possible_download_types:
            return "Download type incorrect. Please check header"

    if download_type=="json":
        mptposts_segmentcount_json=mpt_helper.get_json_all_subjects_segwise_distn(country_code)
        return mptposts_segmentcount_json
    elif download_type=="csv":
        mptposts_segmentcount_df=mpt_helper.get_csv_all_subjects_segwise_distn(country_code)
        resp = make_response(mptposts_segmentcount_df.to_csv(index=False))
        resp.headers["Content-Disposition"] = "attachment; filename="+country_code+"mptposts_timeseries.csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp        


@application.route("/get/mptposts/weekdaywise/<country_code>")
def get_mptposts_weekdaywise(country_code):
    download_type="json"
    possible_download_types=["csv","json"]
    if "download-type" in request.headers:
        print("User requested download type ",request.headers["download-type"])
        download_type=request.headers["download-type"]
        if download_type not in possible_download_types:
            return "Download type incorrect. Please check header"
    if download_type=="json":
        mptposts_weekdaywise_count_json=mpt_helper.get_json_count_weekday_wise(country_code)
        return mptposts_weekdaywise_count_json  
    elif download_type=="csv":
        mptposts_weekdaywise_count_csv=mpt_helper.get_csv_count_weekday_wise(country_code)
        resp = make_response(mptposts_weekdaywise_count_csv.to_csv(index=False))
        resp.headers["Content-Disposition"] = "attachment; filename="+country_code+"mptposts_timeseries.csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp
    else:
        return "No return type specified"





def get_daywise_countplot(country_code):
    df=mpt_helper.get_weekdaywise_count(country_code)
    xs=df["day"]
    ys=df["title"]
    dayWisePlot=mpt_helper.maketime_series_graph(xs,ys)
    return dayWisePlot





@application.route("/get/annualGDP/<country_code>")
# default download type is json
def get_annual_GDP(country_code):
    print("Country code  =",country_code)
    country_code=country_code.upper()
    if country_code not in list_of_country_codes_alpha2:
        return "Incorrect country code"
    download_type="json"
    possible_download_types=["csv","json"]
    if "download-type" in request.headers:
        print("User requested download type ",request.headers["download-type"])
        download_type=request.headers["download-type"]
        if download_type not in possible_download_types:
            return "Download type incorrect. Please check header"

    if download_type=="json":
        annual_GDP_json=helper.get_json_Annual_GDP(country_code)
        return annual_GDP_json
    elif download_type=="csv":
        annual_GDP_df=helper.get_csv_Annual_GDP(country_code)
        resp = make_response(annual_GDP_df.to_csv(index=False))
        resp.headers["Content-Disposition"] = "attachment; filename="+country_code+"gdp.csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp
        


    # default is json
    # unless you have mentioned something in the header

    # get the json data





    return country_code

@application.route('/home')
def homepage():    
    data={}
    return render_template('index.html',data=data)




@application.route("/another")
def notdash2():
    df = pd.DataFrame({
      "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
      "Amount": [4, 1, 2, 2, 4, 5],
      "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
   })
    fig = px.bar(df, x="Fruit", y="Amount", color="City",    barmode="group")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("notdash.html", graphJSON=graphJSON)

@application.route("/")
def notdash():
    
    bar = create_plot()
    # graphJSON=Markup(graphJSON)
    return render_template('notdash.html', plot=bar)


############################################################
# This part for retina image upload for ####################
# diabetes detection #######################################
############################################################
@application.route('/loadretina')
def loadretina():
    return render_template('retina_load_form.html')

@application.route('/retina_image_submit', methods=['GET', 'POST'])
def retina_image_submit():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
         
        path = os.path.join(os.getcwd(),"static","retinopathy", file1.filename)
        file1.save(path)
        diabetes_level=retinopathy_helper.predict_diabetes_level(path)
        # frame = cv2.imread(path)
        # net,layer_names,output_layers,classes=ppl_counter_helper.setupYOLO()
        # frame,count,people_dict=ppl_counter_helper.detect_persons(frame,net,layer_names,output_layers,classes)            
        # print("Number of people in frame = ",count)
        return str(diabetes_level)





############################################################
############################################################
# This part for image upload for person counter ############
############################################################
@application.route('/pplcount')
def ppl_count():
    return render_template('ppl_cntr_form.html')


@application.route('/ppl_counter_submit', methods=['GET', 'POST'])
def upload_file():
    print("pplc")
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
         
        path = os.path.join(os.getcwd(),"static","yolo_ppl_counter", file1.filename)
        file1.save(path)
        frame = cv2.imread(path)
        net,layer_names,output_layers,classes=ppl_counter_helper.setupYOLO()
        frame,count,people_dict=ppl_counter_helper.detect_persons(frame,net,layer_names,output_layers,classes)            
        print("Number of people in frame = ",count)
        return str(count)

    
############################################################
############################################################
# This part for saving google trends data reuest ###########
############################################################
@application.route('/gtrequest')
def gtrequest():
    return render_template('gtrends_request_form.html')

@application.route('/gtrequest_submit',methods=["GET","POST"])
def gtrequest_submit():
    if request.method == 'POST':
        # check which radio button was selected
        dataCount=request.form['DataCountRadio']
        reqemail=request.form['reqemail']
        print(dataCount,reqemail)
        if dataCount=="single":
            # get data individually
            print("Geting a single row")
            searchKeyWord=request.form['searchKeyWord']
            searchType=request.form['searchType']
            fromDate=request.form['fromDate']
            toDate=request.form['toDate']
            regionCode=request.form['regionCode']
            print(searchKeyWord,searchType,fromDate,toDate,regionCode)
            # make a dataframe off this
            data_dic={}
            data_dic["interval"]=[fromDate+" "+toDate]
            data_dic["regionCode"]=[regionCode]            
            data_dic["searchTerm"]=[searchKeyWord]
            if searchType=="category":
                data_dic["isTopic"]=["No"]
                data_dic["isCategory"]=["Yes"]
            elif searchType=="topic":
                data_dic["isTopic"]=["Yes"]
                data_dic["isCategory"]=["No"]
            data_dic["Category"]=[None]
            data_dic["requestorEmail"]=[reqemail]
            print(data_dic)
            df=pd.DataFrame(data_dic)

        elif dataCount=="multiple":
            print("Getting multiple rows")
            if 'file1' not in request.files:
                return 'there is no file1 in form!'
            file1 = request.files['file1']
            df=pd.read_csv(file1)
            df=df.fillna("")

        print("THe data is ")
        # add status and request time
        now = datetime.now()
        df["status"]=["I" for i in range(df.shape[0])]
        df["requestTime"]=[str(now) for i in range(df.shape[0])]
        print(df.head())
        gtrends_helper.update_sheet("googleTrendsRequest",df)

        return "Thank you for submitting the request, results will be sent to your mail"
    return "Stop"








if __name__ == '__main__':
    print(list_of_country_codes_alpha2)

    # run() method of Flask class runs the application
    # on the local development server.
    application.run(host="0.0.0.0",port=5009,debug=True)    