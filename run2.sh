for batchSize in 32 64 128; do
    for learnRate in 0.05 0.1 0.2 0.5; do
        for reLambda in 0 0.001 0.01 0.1; do
            echo ${batchSize} ${learnRate} ${reLambda}
            python mf.py ${batchSize} ${learnRate} ${reLambda} > log/mf_${batchSize}_${learnRate}_${reLambda}.log
            python biasMf.py ${batchSize} ${learnRate} ${reLambda} > log/biasMf_${batchSize}_${learnRate}_${reLambda}.log
        done
    done
done

