#!/bin/bash

for each in astTest/*.h
do 
	#echo "file Name is $each \n"
	a=$each
	echo $a
	python fixClasses.py $each
done