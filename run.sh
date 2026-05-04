echo "============== SEED = 10 ==============\n" 
# alpha = 0.1
echo "SEED=10, ALPHA=0.5, BLOCK1\n"
SPLIT_POINT=block1 PROTOCOL=splitfedv1 ALPHA=0.5 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch
SPLIT_POINT=block1 PROTOCOL=splitfedv2 ALPHA=0.5 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch
echo "SEED=10, ALPHA=0.5, BLOCK2\n"
SPLIT_POINT=block2 PROTOCOL=splitfedv1 ALPHA=0.5 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch
SPLIT_POINT=block2 PROTOCOL=splitfedv2 ALPHA=0.5 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch
echo "SEED=10, ALPHA=0.5, BLOCK3\n"
SPLIT_POINT=block3 PROTOCOL=splitfedv1 ALPHA=0.5 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch
SPLIT_POINT=block3 PROTOCOL=splitfedv2 ALPHA=0.5 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch

# alpha = 0.5
echo "SEED=10, ALPHA=0.1, BLOCK1\n"
SPLIT_POINT=block1 PROTOCOL=splitfedv1 ALPHA=0.1 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch
SPLIT_POINT=block1 PROTOCOL=splitfedv2 ALPHA=0.1 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch
echo "SEED=10, ALPHA=0.1, BLOCK2\n"
SPLIT_POINT=block2 PROTOCOL=splitfedv1 ALPHA=0.1 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch
SPLIT_POINT=block2 PROTOCOL=splitfedv2 ALPHA=0.1 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch
echo "SEED=10, ALPHA=0.1, BLOCK3\n"
SPLIT_POINT=block3 PROTOCOL=splitfedv1 ALPHA=0.1 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch
SPLIT_POINT=block3 PROTOCOL=splitfedv2 ALPHA=0.1 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch

# alpha = 100
echo "SEED=10, ALPHA=100, BLOCK1\n"
SPLIT_POINT=block1 PROTOCOL=splitfedv1 ALPHA=100 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch
SPLIT_POINT=block1 PROTOCOL=splitfedv2 ALPHA=100 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch
echo "SEED=10, ALPHA=100, BLOCK2\n"
SPLIT_POINT=block2 PROTOCOL=splitfedv1 ALPHA=100 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch
SPLIT_POINT=block2 PROTOCOL=splitfedv2 ALPHA=100 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch
echo "SEED=10, ALPHA=100, BLOCK3\n"
SPLIT_POINT=block3 PROTOCOL=splitfedv1 ALPHA=100 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch
SPLIT_POINT=block3 PROTOCOL=splitfedv2 ALPHA=100 SEED=10 ROUNDS=50 sbatch run_creditcard_4client.sbatch

##################### SEEDS = 30
echo "============== SEED = 30 ==============\n"
# alpha = 0.1
echo "SEED=30, ALPHA=0.5, BLOCK1\n"
SPLIT_POINT=block1 PROTOCOL=splitfedv1 ALPHA=0.5 SEED=30 ROUNDS=50 sbatch run_creditcard_4client.sbatch
SPLIT_POINT=block1 PROTOCOL=splitfedv2 ALPHA=0.5 SEED=30 ROUNDS=50 sbatch run_creditcard_4client.sbatch
echo "SEED=30, ALPHA=0.5, BLOCK2\n"
SPLIT_POINT=block2 PROTOCOL=splitfedv1 ALPHA=0.5 SEED=30 ROUNDS=50 sbatch run_creditcard_4client.sbatch
SPLIT_POINT=block2 PROTOCOL=splitfedv2 ALPHA=0.5 SEED=30 ROUNDS=50 sbatch run_creditcard_4client.sbatch
echo "SEED=30, ALPHA=0.5, BLOCK3\n"
SPLIT_POINT=block3 PROTOCOL=splitfedv1 ALPHA=0.5 SEED=30 ROUNDS=50 sbatch run_creditcard_4client.sbatch
SPLIT_POINT=block3 PROTOCOL=splitfedv2 ALPHA=0.5 SEED=30 ROUNDS=50 sbatch run_creditcard_4client.sbatch

# alpha = 0.5
echo "SEED=30, ALPHA=0.1, BLOCK1\n"
SPLIT_POINT=block1 PROTOCOL=splitfedv1 ALPHA=0.1 SEED=30 ROUNDS=50 sbatch run_creditcard_4client.sbatch
SPLIT_POINT=block1 PROTOCOL=splitfedv2 ALPHA=0.1 SEED=30 ROUNDS=50 sbatch run_creditcard_4client.sbatch
echo "SEED=30, ALPHA=0.1, BLOCK2\n"
SPLIT_POINT=block2 PROTOCOL=splitfedv1 ALPHA=0.1 SEED=30 ROUNDS=50 sbatch run_creditcard_4client.sbatch
SPLIT_POINT=block2 PROTOCOL=splitfedv2 ALPHA=0.1 SEED=30 ROUNDS=50 sbatch run_creditcard_4client.sbatch
echo "SEED=30, ALPHA=0.1, BLOCK3\n"
SPLIT_POINT=block3 PROTOCOL=splitfedv1 ALPHA=0.1 SEED=30 ROUNDS=50 sbatch run_creditcard_4client.sbatch
SPLIT_POINT=block3 PROTOCOL=splitfedv2 ALPHA=0.1 SEED=30 ROUNDS=50 sbatch run_creditcard_4client.sbatch

# alpha = 100
echo "SEED=30, ALPHA=100, BLOCK1\n"
SPLIT_POINT=block1 PROTOCOL=splitfedv1 ALPHA=100 SEED=30 ROUNDS=50 sbatch run_creditcard_4client.sbatch
SPLIT_POINT=block1 PROTOCOL=splitfedv2 ALPHA=100 SEED=30 ROUNDS=50 sbatch run_creditcard_4client.sbatch
echo "SEED=30, ALPHA=100, BLOCK2\n"
SPLIT_POINT=block2 PROTOCOL=splitfedv1 ALPHA=100 SEED=30 ROUNDS=50 sbatch run_creditcard_4client.sbatch
SPLIT_POINT=block2 PROTOCOL=splitfedv2 ALPHA=100 SEED=30 ROUNDS=50 sbatch run_creditcard_4client.sbatch
echo "SEED=30, ALPHA=100, BLOCK3\n"
SPLIT_POINT=block3 PROTOCOL=splitfedv1 ALPHA=100 SEED=30 ROUNDS=50 sbatch run_creditcard_4client.sbatch
SPLIT_POINT=block3 PROTOCOL=splitfedv2 ALPHA=100 SEED=30 ROUNDS=50 sbatch run_creditcard_4client.sbatch
echo "============== DONE ==============\n"

