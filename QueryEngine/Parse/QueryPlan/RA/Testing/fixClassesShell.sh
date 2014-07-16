#!/bin/bash

for each in relAlg/*.h
do 
	#echo "file Name is $each \n"
	a=$each
	echo $a
	python fixClasses.py $each
done