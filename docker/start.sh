if [ -z "$1" ]; then
    device=0
else
    device=$1
fi

if [ -z "$2" ]; then
    docker_container_idx=0
else
    docker_container_idx=$2
fi

docker_container_name=lyapunov_rrt_polamp_$docker_container_idx
image_name=lyapunov_rrt_polamp_img

echo "start dockergpu device: $device"
echo "start docker name: $docker_container_name"
echo "start docker image: $image_name"

cd ..
docker run -it --rm --name $docker_container_name --gpus "device=$device" --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -v $(pwd):/usr/home/workspace $image_name "bash"