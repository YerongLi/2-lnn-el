echo "python fill_sim2.py"
python fill_sim2.py
cp ../../../data/aida/template/full_train.csv.2 ../../../data/aida/template/full_train.csv
cp ../../../data/aida/template/full_testA.csv.2 ../../../data/aida/template/full_testA.csv
cp ../../../data/aida/template/full_testB.csv.2 ../../../data/aida/template/full_testB.csv
echo "python fill_context1.py"
python fill_context1.py
cp ../../../data/aida/template/full_train.csv.2 ../../../data/aida/template/full_train.csv
cp ../../../data/aida/template/full_testA.csv.2 ../../../data/aida/template/full_testA.csv
cp ../../../data/aida/template/full_testB.csv.2 ../../../data/aida/template/full_testB.csv
echo "python fill_context2.py"
python fill_context2.py
cp ../../../data/aida/template/full_train.csv.2 ../../../data/aida/template/full_train.csv
cp ../../../data/aida/template/full_testA.csv.2 ../../../data/aida/template/full_testA.csv
cp ../../../data/aida/template/full_testB.csv.2 ../../../data/aida/template/full_testB.csv
echo "python fill_prior.py"
python fill_prior.py
cp ../../../data/aida/template/full_train.csv.2 ../../../data/aida/template/full_train.csv
cp ../../../data/aida/template/full_testA.csv.2 ../../../data/aida/template/full_testA.csv
cp ../../../data/aida/template/full_testB.csv.2 ../../../data/aida/template/full_testB.csv
echo "python fill_dca.py"
python fill_dca.py
cp ../../../data/aida/template/full_train.csv.2 ../../../data/aida/template/full_train.csv
cp ../../../data/aida/template/full_testA.csv.2 ../../../data/aida/template/full_testA.csv
cp ../../../data/aida/template/full_testB.csv.2 ../../../data/aida/template/full_testB.csv