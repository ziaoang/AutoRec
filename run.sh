for batchSize in 32 64 128; do
    for learnRate in 0.01 0.05 0.1 0.5; do
        echo ${batchSize} ${learnRate}
        python concat_a_b.py ${batchSize} ${learnRate} > log/a_b_${batchSize}_${learnRate}.log
        python concat_a_b_ab.py ${batchSize} ${learnRate} > log/a_b_ab_${batchSize}_${learnRate}.log
        python concat_ab.py ${batchSize} ${learnRate} > log/ab_${batchSize}_${learnRate}.log
    done
done

for batchSize in 32 64 128; do
    for learnRate in 0.01 0.05 0.1 0.5; do
        for reLambda in 0 0.01 0.1; do
            echo ${batchSize} ${learnRate} ${reLambda}
            python biasMf.py ${batchSize} ${learnRate} ${reLambda} > log/biasMf_${batchSize}_${learnRate}_${reLambda}.log
        done
    done
done

