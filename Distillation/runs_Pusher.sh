



alphas=(0.0)
datasets=('50k' '10k')

Modes=('ImpaalaSmall' 'ImpaalaMid' 'ImpaalaBig')
Modes=('Impaala')

GPU_ID=2
MAX_JOBS=13  # Numero massimo di job in parallelo

# Funzione per aspettare se ci sono troppi job
wait_for_slot() {
    while [ $(jobs -rp | wc -l) -ge $MAX_JOBS ]; do
        sleep 10
    done
}

# Crea lista di comandi
commands=()
for mode in "${Modes[@]}"
do
    commands+=("python -m Distillation.Pusher --Sdistillation --mode $mode --device cuda:2")
    commands+=("python -m Distillation.Pusher --Tdistillation --mode $mode --device cuda:2")

    for PPD_parameter in 5
    do
        commands+=("python -m Distillation.Pusher --PPD --PPD_parameter $PPD_parameter --mode $mode --device cuda:2")
    done
    
    for dataset in "${datasets[@]}"
    do
        for alpha in "${alphas[@]}"
        do
            commands+=("python -m Distillation.Pusher --BC_phase --mode $mode --dataset $dataset --alpha $alpha --device cuda:1 ")
        done
    done
done

# Lancia i comandi con max MAX_JOBS in parallelo
for cmd in "${commands[@]}"; do
    wait_for_slot
    echo "Launching: $cmd"
    eval "$cmd" &
done

wait  # Aspetta che tutti i job finiscano
echo "All jobs completed!"

# pkill -f "python -m Test.Pusher"
