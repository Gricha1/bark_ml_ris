cd ..
# check if MFNLC_for_polamp_env exist
if [ -z "$(ls -A MFNLC_for_polamp_env)" ]; then
   echo "MFNLC_for_polamp_env folder doesnt exist!!!"
else
   docker build -t lyapunov_rrt_polamp_img -f docker/dockerfile .
fi
