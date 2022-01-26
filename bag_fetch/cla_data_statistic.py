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
from pymdi import pymdi
import argparse

def bag_statistic(bag_type):
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
                     {"timestamp": {"gte":"1641802032000", "lte": "1642838832000"}}
                ]
            }
        }, limit=10000000
    )
    data, total = result["data"], result["total"]
    metas = client.get_meta([x["md5"] for x in data])
    class_map = {}

    for mem in metas:
        name_spt = mem['name'].split('_')
        if 'no-sensor.bag' in name_spt:
            continue
        class_feature = []
        start = name_spt.index('event')+1
        end = name_spt.index('filter')
        for i in range(start,end):
            class_feature.append(name_spt[i])
        class_str = '_'.join(class_feature)
        if class_str in class_map.keys():
            class_map[class_str] += 1
        else:
            class_map[class_str] = 1

    print('-'*80)
    for item in class_map.items():
        print(f'{item}\n')
    print('-'*80)
    print(f'The number of data is : {total}')
    print(f'There are  -> {len(class_map)} <- classes')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data_download')
    parser.add_argument('--bag_type', '-t', required=True, help='bag type needs to be counted')
    args = parser.parse_args()
    bag_type = args.bag_type
    bag_statistic(bag_type)
    




    

    
