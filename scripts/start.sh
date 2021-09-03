for i in {2..9}
do
	ssh hduser@hadoop${i} "export PYTHONPATH=$PYTHONPATH:/home/hduser/@ddl/;cd ~/@ddl;nohup python3.8 ddltrain2/worker.py ${i} hadoop${i} 9 > /dev/null 2>&1 &"
  #ssh hduser@hadoop${i} "export PYTHONPATH=$PYTHONPATH:/home/hduser/@ddl/;cd ~/@ddl;nohup python3.8 ddltrain2/worker.py ${i} hadoop${i} 9 2> /dev/null &"
  echo "hadoop${i} is already..."
done