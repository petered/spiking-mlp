virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
printf '\nexport PYTHONPATH=$PYTHONPATH:%s\n' $PWD >> venv/bin/activate
#echo "./../../../src/deepstream/src/main/python/" > ./venv/lib/python2.7/site-packages/deepstream.pth
echo "Spiking-MLP Setup Complete! (As long as there're no errors above)"
