## ==== Download and preprocessing
DOCKER_OUTPUT_PATH=/home/saps/output
IMAGES_DIR=./input

## Note: The following variables are used to run the docker containers and are not used in the Makefile itself
#  		 	 The container only retrieves images from between 1984 and 2017. Furthermore, not every day has  
#			 	 images of a specific region because the satellite in orbit collects images every 3 days.
# Samples:
# | landsat_8  | landsat_5  | landsat_7
# | 215065     | 215065     | 215065
# | 2020-10-10 | 1990-03-14 | 2000-12-30
IMAGE_LANDSAT="landsat_8"
IMAGE_PATHROW="215065"
IMAGE_DATE="2020-10-10"

## ==== GPU architecture
ARCH=sm_86

## ==== Docker configuration
DOCKER_IMAGE_NAME=landsat-evapo
DOCKER_GPU_ARGS=--gpus all

## ==== Execution
METHOD=0
THREADS=64
OUTPUT_DATA_PATH=./output
INPUT_DATA_PATH=$(IMAGES_DIR)/$(IMAGE_LANDSAT)_$(IMAGE_PATHROW)_$(IMAGE_DATE)/final_results

clean:
	rm $(OUTPUT_DATA_PATH)/*

clean-all:
	rm -rf $(OUTPUT_DATA_PATH)/*

build:
	nvcc -arch=$(ARCH) -I ./include -O3 --use_fast_math -g ./src/*.cu -o ./main -std=c++14 -ltiff -rdc=true

fix-permissions:
	sudo chmod -R 755 $(INPUT_DATA_PATH)/*

exec-landsat8:
	./bin/run-exp.sh \
		$(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF $(INPUT_DATA_PATH)/B4.TIF \
		$(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF $(INPUT_DATA_PATH)/B.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) -threads=$(THREADS) & 

exec-landsat5-7:
	./bin/run-exp.sh \
		$(INPUT_DATA_PATH)/B1.TIF $(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF \
		$(INPUT_DATA_PATH)/B4.TIF $(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) -threads=$(THREADS) &

nsys-landsat8:
	./bin/run-nsys.sh \
		$(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF $(INPUT_DATA_PATH)/B4.TIF \
		$(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF $(INPUT_DATA_PATH)/B.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) -threads=$(THREADS) & 

nsys-landsat5-7:
	./bin/run-nsys.sh \
		$(INPUT_DATA_PATH)/B1.TIF $(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF \
		$(INPUT_DATA_PATH)/B4.TIF $(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) -threads=$(THREADS) &

ncu-landsat5-7:
	./bin/run-ncu.sh \
		$(INPUT_DATA_PATH)/B1.TIF $(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF \
		$(INPUT_DATA_PATH)/B4.TIF $(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) -threads=$(THREADS) &

ncu-landsat8:
	./bin/run-ncu.sh \
		$(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF $(INPUT_DATA_PATH)/B4.TIF \
		$(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF $(INPUT_DATA_PATH)/B.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) -threads=$(THREADS) & 

analisys-landsat8:
	./bin/run-ana.sh \
		$(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF $(INPUT_DATA_PATH)/B4.TIF \
		$(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF $(INPUT_DATA_PATH)/B.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) -threads=$(THREADS) &

analisys-landsat5-7:
	./bin/run-ana.sh \
		$(INPUT_DATA_PATH)/B1.TIF $(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF \
		$(INPUT_DATA_PATH)/B4.TIF $(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) -threads=$(THREADS) &

## ==== Docker targets

docker-build:
	docker build -t $(DOCKER_IMAGE_NAME) --build-arg CUDA_ARCH=$(ARCH) .

docker-test-nvidia:
	docker run --rm $(DOCKER_GPU_ARGS) nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

docker-run-landsat8: 
	docker run $(DOCKER_GPU_ARGS) \
		-v $(shell pwd)/$(IMAGES_DIR):/app/$(IMAGES_DIR) \
		-v $(shell pwd)/$(OUTPUT_DATA_PATH):/app/$(OUTPUT_DATA_PATH) \
		$(DOCKER_IMAGE_NAME) \
		$(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF $(INPUT_DATA_PATH)/B4.TIF \
		$(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF $(INPUT_DATA_PATH)/B.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) -threads=$(THREADS)

docker-run-landsat5-7: 
	docker run $(DOCKER_GPU_ARGS) \
		-v $(shell pwd)/$(IMAGES_DIR):/app/$(IMAGES_DIR) \
		-v $(shell pwd)/$(OUTPUT_DATA_PATH):/app/$(OUTPUT_DATA_PATH) \
		$(DOCKER_IMAGE_NAME) \
		$(INPUT_DATA_PATH)/B1.TIF $(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF \
		$(INPUT_DATA_PATH)/B4.TIF $(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) -threads=$(THREADS)

docker-landsat-download:
	docker run \
		-v $(IMAGES_DIR):$(DOCKER_OUTPUT_PATH) \
		-e OUTPUT_PATH=$(DOCKER_OUTPUT_PATH) \
		-e LANDSAT=$(IMAGE_LANDSAT) \
		-e PATHROW=$(IMAGE_PATHROW) \
		-e DATE=$(IMAGE_DATE) \
		cilasmarques/landsat-download:latest

docker-landsat-preprocess:
	docker run \
		-v $(IMAGES_DIR):$(DOCKER_OUTPUT_PATH) \
		-e OUTPUT_PATH=$(DOCKER_OUTPUT_PATH) \
		-e LANDSAT=$(IMAGE_LANDSAT) \
		-e PATHROW=$(IMAGE_PATHROW) \
		-e DATE=$(IMAGE_DATE) \
		cilasmarques/landsat-preprocess:latest

