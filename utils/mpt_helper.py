from flask import jsonify
import requests


import time

import pandas as pd
import os

import pickle
import requests
from dateutil.parser import parse
import datetime
import matplotlib.pyplot as plt

import numpy as np
import json
import requests

import plotly.express as px
import plotly

def maketime_series_graph(xs,ys):
    
    fig = px.line( x=xs, y=ys, title='Daily postings for help in education')

    fig.update_layout(
        title="Daily postings for help in education",
        xaxis_title="Date",
        yaxis_title="Number of postings",
    #     legend_title="Legend Title",
    #     font=dict(
    #         family="Courier New, monospace",
    #         size=18,
    #         color="RebeccaPurple"
    #     )
    )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON
    

def make_grouped_bar_chart(xs,ys,zs):

    fig = px.bar( x=xs, y=ys,color=zs)

    fig.update_layout(
        title="Popularity of Different Segments",
        xaxis_title="Segment Name",
        yaxis_title="Number of postings in last 60 days",
    #     legend_title="Legend Title",
    #     font=dict(
    #         family="Courier New, monospace",
    #         size=18,
    #         color="RebeccaPurple"
    #     )
    )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON




def make_bar_chart(xs,ys,the_title=None,xaxis_title=None):
    fig = px.bar( x=xs, y=ys)
    if the_title==None:
        the_title="Popularity of Different Segments"
    if xaxis_title==None:
        xaxis_title="Segment Name"
    fig.update_layout(
        title=the_title,
        xaxis_title=xaxis_title,
        yaxis_title="Number of postings in last 60 days",
    #     legend_title="Legend Title",
    #     font=dict(
    #         family="Courier New, monospace",
    #         size=18,
    #         color="RebeccaPurple"
    #     )
    )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON



def get_csv_all_subjects_segwise_distn(country_code):
    df=pd.read_csv("mpt_data.csv")
    df_group_segment=df.groupby(["segment"]).count().reset_index()
    df_group_segment=df_group_segment.sort_values(["title"],ascending=False)
    uniq_segments_ordered=df_group_segment["segment"]
    dic_seg_subjects={}
    dic_seg_subjects["segment"]=[]
    dic_seg_subjects["subject"]=[]
    dic_seg_subjects["count"]=[]
    for uniq_segment in uniq_segments_ordered:
        df_segment_wise=df[df["segment"]==uniq_segment]
        df_segment_wise_group_subject=df_segment_wise.groupby(["subject"]).count().reset_index()
        df_segment_wise_group_subject=df_segment_wise_group_subject.sort_values(["title"],ascending=False)
        dic_seg_subjects["subject"].extend(df_segment_wise_group_subject["subject"])
        dic_seg_subjects["count"].extend(df_segment_wise_group_subject["title"])
        dic_seg_subjects["segment"].extend([uniq_segment for i in range(df_segment_wise_group_subject.shape[0])])

    df=pd.DataFrame(dic_seg_subjects)
    
    return df

def get_json_all_subjects_segwise_distn(country_code):
    df=pd.read_csv("mpt_data.csv")
    df_group_segment=df.groupby(["segment"]).count().reset_index()
    df_group_segment=df_group_segment.sort_values(["title"],ascending=False)
    uniq_segments_ordered=df_group_segment["segment"]
    dic_seg_subjects={}
    for uniq_segment in uniq_segments_ordered:
        if uniq_segment not in dic_seg_subjects:
            dic_seg_subjects[uniq_segment]={}
        df_segment_wise=df[df["segment"]==uniq_segment]
        df_segment_wise_group_subject=df_segment_wise.groupby(["subject"]).count().reset_index()
        df_segment_wise_group_subject=df_segment_wise_group_subject.sort_values(["title"],ascending=False)
        for index,row in df_segment_wise_group_subject.iterrows():
            dic_seg_subjects[uniq_segment][row["subject"]]=row["title"]
    return dic_seg_subjects


def get_plot_show_mptposts_subjects_per_segment(country_code):
    '''
    this function gets all distributions
    of subjetcs in segments
    creates all the plots and returns
    '''
    df=get_csv_all_subjects_segwise_distn(country_code)
    '''
    columns are subject, count, segment
    '''
    segments_unique=df["segment"].unique()
    plot={}
    plot["subjectspersegment"]={}
    plot["subjectspersegment"]["plots"]=[]
    for segment in segments_unique:
        df_segment=df[df["segment"]==segment]
        xs=df_segment["subject"]
        ys=df_segment["count"]
        the_title="Requirements posted for "+segment
        xaxis_title="Subjects in "+segment
        barPlot=make_bar_chart(xs,ys,the_title=the_title,xaxis_title=xaxis_title)
        plot["subjectspersegment"]["plots"].append(barPlot)
    plot["subjectspersegment"]["count"]=len(plot["subjectspersegment"]["plots"])    
    
    return plot



def get_time_series_posts(country_code):
    df=pd.read_csv("mpt_data.csv")
    df["date_posted"]=pd.to_datetime(df["date_posted"],format="%d/%m/%y")
    df["date_posted"]=df["date_posted"].dt.date
    df_group_dates=df.groupby(["date_posted"]).count().reset_index()
    df_group_dates=df_group_dates.sort_values(["date_posted"],ascending=False)
    # return xvalues yvalues
    
    return list(df_group_dates["date_posted"]),list(df_group_dates["title"])

def get_json_mptposts_timeseries(country_code):
    dates,counts=get_time_series_posts(country_code)
    dic={}
    for i in range(len(dates)):
        dic[str(dates[i])]=counts[i]
    return dic

def get_csv_mptposts_timeseries(country_code):
    dates,counts=get_time_series_posts(country_code)
    dic={}
    dic["date_posted"]=dates
    dic["count"]=counts
    df=pd.DataFrame(dic)
    
    return df



def get_segment_count(country_code):
    df=pd.read_csv("mpt_data.csv")
    df_group_segment=df.groupby(["segment"]).count().reset_index()
    df_group_segment=df_group_segment.sort_values(["title"],ascending=False)    
    return list(df_group_segment['segment']),list(df_group_segment['title'])

def get_json_segmentcount(country_code):
    seg_names,counts=get_segment_count(country_code)
    dic={}
    for i in range(len(seg_names)):
        dic[str(seg_names[i])]=counts[i]
    return dic



def get_csv_segmentcount(country_code):
    seg_names,counts=get_segment_count(country_code)
    dic={}
    dic["segments"]=seg_names
    dic["count"]=counts
    df=pd.DataFrame(dic)
    
    return df



def get_grouped_segment_subject(country_code):
    df=pd.read_csv("mpt_data.csv")
    segment=df['segment']
    title=df['title']
    subject=df["subject"]

    return segment, title, subject

def get_json_count_weekday_wise(country_code):
	df_group_days=get_weekdaywise_count(country_code)

	dic_data={}
	for index,row in df_group_days.iterrows():
		dic_data[row["day"]]=row["title"]

	return dic_data

def get_csv_count_weekday_wise(country_code):
	
	df_group_days=get_weekdaywise_count(country_code)
	dic_data={}
	dic_data["day"]=[]
	dic_data["count"]=[]
	for index,row in df_group_days.iterrows():
		dic_data["day"].append(row["day"])
		dic_data["count"].append(row["title"])
	df=pd.DataFrame(dic_data)
	return df


def get_weekdaywise_count(country_code):
	df=pd.read_csv("mpt_data.csv")
	df["date_posted"]=pd.to_datetime(df["date_posted"],format="%d/%m/%y")
	df["day"]=df["date_posted"].dt.dayofweek
	df_group_days=df.groupby(["day"]).count().reset_index()

	dict_num_days={
	    0:"Mon",
	    1:"Tues",
	    2:"Wed",
	    3:"Thurs",    
	    4:"Fri",    
	    5:"Sat",    
	    6:"Sun",    
	    
	}
	df_group_days["day"]=df_group_days["day"].replace(dict_num_days)
	return df_group_days



