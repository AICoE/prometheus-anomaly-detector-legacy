# Required Variables
bearer_token=
prometheus_url=

block_storage_access_key=
block_storage_secret_key=
block_storage_bucket_name=
block_storage_endpoint_url=

# Optional Variables
oc_app_name=train-prom-dh-prod
docker_app_name=train-prometheus

docker_build:
	docker build -t ${docker_app_name} .

docker_test:
	docker run ${docker_app_name}

docker_run:
	docker run -ti --rm \
	   --env "BEARER_TOKEN=${bearer_token}" \
	   --env "URL=${prometheus_url}" \
		 --env BOTO_ACCESS_KEY="${block_storage_access_key}" \
		 --env BOTO_SECRET_KEY="${block_storage_secret_key}" \
		 --env BOTO_OBJECT_STORE="${block_storage_bucket_name}" \
		 --env BOTO_STORE_ENDPOINT="${block_storage_endpoint_url}" \
	   ${docker_app_name}:latest

oc_deploy:
	oc new-app --file=./train-prophet-deployment-template.yaml --param APPLICATION_NAME="${oc_app_name}" \
			--param URL="${prometheus_url}" \
			--param BEARER_TOKEN="${bearer_token}" \
			--param BOTO_ACCESS_KEY="${block_storage_access_key}" \
			--param BOTO_SECRET_KEY="${block_storage_secret_key}" \
			--param BOTO_OBJECT_STORE="${block_storage_bucket_name}" \
			--param BOTO_STORE_ENDPOINT="${block_storage_endpoint_url}"

oc_delete_all:
	oc delete all -l app=${oc_app_name}

run_model:
	BEARER_TOKEN=${bearer_token} \
	URL=${prometheus_url} \
	BOTO_ACCESS_KEY=${block_storage_access_key} \
	BOTO_SECRET_KEY=${block_storage_secret_key} \
	BOTO_OBJECT_STORE=${block_storage_bucket_name} \
	BOTO_STORE_ENDPOINT=${block_storage_endpoint_url} \
	python3 ../train-prometheus-prod/train-prometheus/app.py
