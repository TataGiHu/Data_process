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

# super parameters
BAG_TYPE=HNP_wm_cruising
# dir mark
EXTRACTED_MARK=_extracted
GENERATE_MARK=_train_data

echo "<<<<<<<<<   Downloading  <<<<<<<<<<"
python3 ./bag_download.py \
        -t $BAG_TYPE 
#python3 ./bag_download.py -t $BAG_TYPE -d $BAG_DIR -n 100

source /opt/ros/melodic/setup.sh
echo "<<<<<<<<<    Extracting  <<<<<<<<<<"
mkdir ../../$BAG_TYPE$EXTRACTED_MARK
python2 ../extract_bag.py \
        -b ../../$BAG_TYPE \
        -s ../../$BAG_TYPE$EXTRACTED_MARK

source /opt/ros/melodic/setup.sh
echo "<<<<<<<<<    Generating  <<<<<<<<<<"
mkdir ../../$BAG_TYPE$GENERATE_MARK
python2 ../generate_train_data_points.py \
        -b ../../$BAG_TYPE$EXTRACTED_MARK \
        -s ../../$BAG_TYPE$GENERATE_MARK

echo "<<<<<<<<<    Transfering   <<<<<<<<<"
cp -r ../../$BAG_TYPE$GENERATE_MARK ../../../yupeng
echo "<<<<<<<<<    Finished   <<<<<<<<<"