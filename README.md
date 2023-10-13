# RandParcel

To run the whole experiment, please run

```
$ chmod +x run_loop_main.sh
$ ./run_loop_main.sh
```


To run a single clique function, please run

```
$ python gpt_efficiency_main.py --dataset hotpot_qa --model gpt-3.5-turbo --mode avg_length
```

If you want to resume the previous experiment, please run

```
$ python gpt_efficiency_main.py --dataset trec --model gpt-4 --mode random --resume True
```

