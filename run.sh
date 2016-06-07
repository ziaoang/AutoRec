for learnRate in 0.1 0.2 0.5; do
    for batchSize in 64; do
        for regular in 0 0.1 0.01; do
            echo ${learnRate} ${batchSize} ${regular}
            python mf.py ${learnRate} ${batchSize} ${regular} > log/${learnRate}_${batchSize}_${regular}.log
        done
    done
done
