SHELLNAME=bash#/bin/zsh

install:
	pip3 install -e .

docker:
	docker build --network host -f Dockerfile -t inclearn_docker:latest .
	docker run -it --runtime=nvidia --name inclearn_${USER}${SUFFIX} -v /srv/data/:/srv/data/ -v /tmp:/tmp -v /home/${USER}:/workspace -p 10.0.4.4:"${PORT}":8888 --entrypoint ${SHELLNAME} inclearn_docker:latest

join:
	docker exec -it inclearn_${USER}${SUFFIX} ${SHELLNAME}

jupy:
	jupyter-notebook --no-browser --allow-root --port=8891

tests:
	pytest -p no:cacheprovider tests/


.PHONY: tests
