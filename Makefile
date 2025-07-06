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

## ==== Execution
METHOD=0
OUTPUT_DATA_PATH=./output
INPUT_DATA_PATH=$(IMAGES_DIR)/$(IMAGE_LANDSAT)_$(IMAGE_PATHROW)_$(IMAGE_DATE)/final_results

## ==== Docker
DOCKER_IMAGE_NAME=landsat-evapo
DOCKER_CONTAINER_NAME=landsat-evapo-container

clean:
	rm $(OUTPUT_DATA_PATH)/*

clean-all:
	rm -rf $(OUTPUT_DATA_PATH)/*

build:
	g++ -I./include -g ./src/*.cpp -o ./main -std=c++14 -ltiff -pthread

fix-permissions:
	sudo chmod -R 755 $(INPUT_DATA_PATH)/*

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

exec-landsat8:
	./bin/run-exp.sh \
		$(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF $(INPUT_DATA_PATH)/B4.TIF \
		$(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF $(INPUT_DATA_PATH)/B.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) & 

exec-landsat5-7:
	./bin/run-exp.sh \
		$(INPUT_DATA_PATH)/B1.TIF $(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF \
		$(INPUT_DATA_PATH)/B4.TIF $(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) &

analisys-landsat8:
	./bin/run-ana.sh \
		$(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF $(INPUT_DATA_PATH)/B4.TIF \
		$(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF $(INPUT_DATA_PATH)/B.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) &

analisys-landsat5-7:
	./bin/run-ana.sh \
		$(INPUT_DATA_PATH)/B1.TIF $(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF \
		$(INPUT_DATA_PATH)/B4.TIF $(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD) &

docker-build:
	docker build -t $(DOCKER_IMAGE_NAME) .

docker-run:
	docker run --rm -it \
		-v $(PWD)/input:/app/input \
		-v $(PWD)/output:/app/output \
		--name $(DOCKER_CONTAINER_NAME) \
		$(DOCKER_IMAGE_NAME)

docker-run-landsat8:
	docker run --rm -it \
		-v $(PWD)/input:/app/input \
		-v $(PWD)/output:/app/output \
		--name $(DOCKER_CONTAINER_NAME) \
		$(DOCKER_IMAGE_NAME) \
		./bin/run-exp.sh \
		$(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF $(INPUT_DATA_PATH)/B4.TIF \
		$(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF $(INPUT_DATA_PATH)/B.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD)

docker-run-landsat5-7:
	docker run --rm -it \
		-v $(PWD)/input:/app/input \
		-v $(PWD)/output:/app/output \
		--name $(DOCKER_CONTAINER_NAME) \
		$(DOCKER_IMAGE_NAME) \
		./bin/run-exp.sh \
		$(INPUT_DATA_PATH)/B1.TIF $(INPUT_DATA_PATH)/B2.TIF $(INPUT_DATA_PATH)/B3.TIF \
		$(INPUT_DATA_PATH)/B4.TIF $(INPUT_DATA_PATH)/B5.TIF $(INPUT_DATA_PATH)/B6.TIF \
		$(INPUT_DATA_PATH)/B7.TIF $(INPUT_DATA_PATH)/elevation.tif $(INPUT_DATA_PATH)/MTL.txt \
		$(INPUT_DATA_PATH)/station.csv $(OUTPUT_DATA_PATH) \
		-meth=$(METHOD)

docker-clean:
	docker rmi $(DOCKER_IMAGE_NAME) 2>/dev/null || true
	docker container prune -f
