WORKING_DIR=.
ENV_NAME=playground

all:
	python setup.py sdist bdist_wheel

clean:
	rm -rf ./build
	rm -rf ./dist

upload_test: clean all
	twine upload --repository testpypi dist/*

upload: clean all
	twine upload dist/*

