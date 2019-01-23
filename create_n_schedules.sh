#!/bin/bash
echo 'How many schedules do you want?'
read num_schedules
for ((i=1; i <=$num_schedules;i++))
do
 python create_data.py
done

