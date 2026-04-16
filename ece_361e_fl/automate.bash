startexp=18
endexp=25

i=$startexp
while [ $i -le $endexp ]; do
    venv/Scripts/python.exe generate_figs.py --exps $i --exp_labels IID --runs 1 --plot_title Exp$i --accuracy --time --power --energy --communication
    i=$((i + 1))
done