# -*- coding: utf-8 -*-
# @Author  : Ta_Mingquan
# @File    : $bad_download_and_screen.py
# @Function: Download the expected data bags in cla and screen them to meet the need
# @Return  : .bag files
# @Note    : "token" value eference：https://confluence.momenta.works/pages/viewpage.action?spaceKey=TD&title=SDK
#            "must" values reference：https://confluence.momenta.works/display/TD/Search+API+Sample
# @Must    : install pymdi ：pip3 install --extra-index-url https://artifactory.momenta.works/artifactory/api/pypi/pypi-pl/simple pymdi==0.1.11

#####################################################################
#      Scene Classification Reference
#-------------------------------------------------------------------
#    cruising： *HNP_wm_cruising*
#    curve： *HNP_wm_sharp_turning* / *HNP_wm_ssharp_turning*
#    diversion： *HNP_wm_ls_con*
#    confluence： *HNP_wm_lm_n_c* / *HNP_wm_lm_is_c*
#    intersection： *HNP_wm_in_int*
#    lane change： *HNP_wm_lc_clane*
#    ramp: *HNP_wm_in_ramp*
#--------------------
#    all wm bag: *wm*
#####################################################################
import os
import argparse
from pydoc_data import topics
from pymdi import pymdi
from time import *


def data_download(bag_type, download_dir, bag_num):
    bag_type_wildcard = '*' + bag_type + '*'
    client = pymdi.Client(token="618b20c3-f494-4a12-808c-e88fd80db117")
  
    # get data by searching, download data and relative
    result = client.query(
        {
            "bool": {
                "must": [
                     {"tag":"processed_by_mpilot-highway-1-3-3-dxp"},
                     {"type":"bag"},
                     {"wildcard":{"name":bag_type_wildcard}},
                     # Epoch & Unix timestemp conversion tools: https://www.epochconverter.com/
                     {"timestamp": {"gte":"1641830399000", "lte": "1642780799000"}}
                ]
            }
        }, limit=10000000
    )
    data, total = result["data"], result["total"]
    metas = client.get_meta([x["md5"] for x in data])
    if not download_dir:
        # make a saving directory
        os.system("mkdir ../../" + bag_type)
        bag_root = "../../" + bag_type
    else:
        bag_root = download_dir
    # download bags
    print('---- Downloading Start ----')
    bag_count = 0
    
    for mem in metas:
        if os.path.exists(bag_root + '/' + mem['name']):
            continue
        name_spt = mem['name'].split('_')
        if 'no-sensor.bag' in name_spt:
            continue
        client.download_to_dir(mem, bag_root)
        bag_count += 1
        if bag_num and bag_count >= bag_num:
            break
    print('---- Downloading Complished ----')

   


    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='data_download')
    parser.add_argument('--bag_type', '-t', required=True, help='bag type needs to be downloaded')
    parser.add_argument('--download_dir', '-d', required=False, help='directory to save bags')
    parser.add_argument('--bag_num', '-n', required=False, type=int, help='expecting number of bags')
    args = parser.parse_args()
    bag_type = args.bag_type
    download_dir = args.download_dir
    bag_num = args.bag_num
    data_download(bag_type, download_dir, bag_num)
    
    
    


    

    
