from flask import Flask, render_template, request,send_from_directory, jsonify, url_for
from flask import redirect, make_response
import plotly
import plotly.graph_objs as go
import numpy as np
import pycountry

import pandas as pd
import os
import cv2

from utils import helper
from utils import mpt_helper
from utils import ppl_counter_helper

import json
import plotly.express as px
from flask import Markup

def generate_country_codes():
    list_of_country_codes_alpha2=[]
    for country in list(pycountry.countries):
#     print(country.alpha_2)
        list_of_country_codes_alpha2.append(country.alpha_2)
    return list_of_country_codes_alpha2



list_of_country_codes_alpha2=generate_country_codes()
# will contain the list of country codes 2 alphabets


application = Flask(__name__)

def create_yearwise_gdp_plot(x_vals,y_vals,country):
    print(type(x_vals[0]))
    print(type(y_vals[0]))
    
    data = [
        go.Scatter(
            x=x_vals, # assign x as the dataframe column 'x'
            y=y_vals            
        )
    ]

    fig = go.Figure(data=data)
 
    # title alignment
    fig.update_layout(title_text='GDP of '+country+" across years")


    # graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

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


@application.route("/show/mptposts/timeseries/<country_code>")
def show_mptposts_timeseries(country_code):
    xs,ys=mpt_helper.get_time_series_posts(country_code)
    xs.reverse()
    ys.reverse()
    timeSeriesPlot=mpt_helper.maketime_series_graph(xs,ys)
    plot={}
    plot["timeSeries"]=timeSeriesPlot

    return render_template("plot_mpt.html", plot=plot)


@application.route("/show/mptposts/segmentcount/<country_code>")
def show_mptposts_segmentcount(country_code):
    xs,ys=mpt_helper.get_segment_count(country_code)
    # xs.reverse()
    # ys.reverse()
    barPlot=mpt_helper.make_bar_chart(xs,ys)
    plot={}
    plot["segmentBarGraph"]=barPlot

    return render_template("plot_mpt.html", plot=plot)    


@application.route("/show/mptposts/groupedsegmentsubject/<country_code>")
def show_mptposts_grouped_segment_subject(country_code):
    '''
    colorful plot showing all subjects per segment in
    1 graph
    '''
    xs,ys,zs=mpt_helper.get_grouped_segment_subject(country_code)    
    barPlot=mpt_helper.make_grouped_bar_chart(xs,ys,zs)
    plot={}
    plot["groupedSegmentBarGraph"]=barPlot

    return render_template("plot_mpt.html", plot=plot)    



def get_daywise_countplot(country_code):
    df=mpt_helper.get_weekdaywise_count(country_code)
    xs=df["day"]
    ys=df["title"]
    dayWisePlot=mpt_helper.maketime_series_graph(xs,ys)
    return dayWisePlot

@application.route("/show/mptposts/weekdaywise/<country_code>")
def show_mptposts_weekdaywise(country_code):   
    print("at show_mptposts_weekdaywise function") 
    plot={}
    plot["dayWisePlot"]=get_daywise_countplot(country_code)

    return render_template("plot_mpt.html", plot=plot)





@application.route("/show/mptposts/subjectspersegment/<country_code>")
def show_mptposts_subjects_per_segment(country_code):
    '''
    multiple plots showing all subjects per segment in
    multiple graph
    '''
    plot=mpt_helper.get_plot_show_mptposts_subjects_per_segment(country_code)
    '''
    gives a dict
    dic:
        subjectspersegment:
            plots: [plot1, plot2, ...]
            count: number


    '''

    return render_template("plot_mpt.html", plot=plot)    





@application.route("/show/mptposts/allplots/<country_code>")
def show_mptposts_allplots(country_code):
    plot={}

    xs,ys=mpt_helper.get_time_series_posts(country_code)
    xs.reverse()
    ys.reverse()
    timeSeriesPlot=mpt_helper.maketime_series_graph(xs,ys)
    
    plot["timeSeries"]=timeSeriesPlot


    xs,ys=mpt_helper.get_segment_count(country_code)    
    barPlot=mpt_helper.make_bar_chart(xs,ys)
    plot["segmentBarGraph"]=barPlot


    xs,ys,zs=mpt_helper.get_grouped_segment_subject(country_code)    
    barPlot=mpt_helper.make_grouped_bar_chart(xs,ys,zs)
    plot["groupedSegmentBarGraph"]=barPlot


    plot_subject_segmentwise=mpt_helper.get_plot_show_mptposts_subjects_per_segment(country_code)
    '''
    gives a dict
    dic:
        subjectspersegment:
            plots: [plot1, plot2, ...]
            count: number
    '''
    plot["subjectspersegment"]=plot_subject_segmentwise["subjectspersegment"]


    # this one for weekday count
    plot["dayWisePlot"]=get_daywise_countplot(country_code)


    return render_template("plot_mpt.html", plot=plot)    





@application.route("/show/annualGDP/<country_code>")
def show_annual_GDP(country_code):
    country_code=country_code.upper()
    print("Country code  =",country_code)
    if country_code not in list_of_country_codes_alpha2:
        return "Incorrect country code"
    country_name=pycountry.countries.get(alpha_2=country_code).name
    annual_GDP_df=helper.get_csv_Annual_GDP(country_code)
    print(annual_GDP_df.head())
    annual_GDP_df = annual_GDP_df.astype({"annualGDP": float})

    years=list(annual_GDP_df["year"])
    years.reverse()
    valsf=list(annual_GDP_df["annualGDP"])    
    valsf.reverse()
    plot=create_yearwise_gdp_plot(years,valsf,country_name)
    return render_template("plot.html", plot=plot)
    




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


def create_plot():
    N = 40
    x = np.linspace(0, 1, N)
    y = np.random.randn(N)
    df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe

    data = [
        go.Bar(
            x=df['x'], # assign x as the dataframe column 'x'
            y=df['y']
        )
    ]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

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
############################################################
# This part for image upload for person counter ############
############################################################
@application.route('/pplcount')
def ppl_count():
    return render_template('ppl_cntr_form.html')


@application.route('/ppl_counter_submit', methods=['GET', 'POST'])
def upload_file():
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

    




if __name__ == '__main__':
    print(list_of_country_codes_alpha2)
    ppl_counter_helper.setupYOLO()
    # run() method of Flask class runs the application
    # on the local development server.
    application.run(host="0.0.0.0",port=5009,debug=True)    