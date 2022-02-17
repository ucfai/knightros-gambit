# example use:
# ./train.sh streamlit
# ./train.sh json

if [ "$1" = "streamlit" ]; then
    streamlit run $2.py
elif [ "$1" = "json" ]; then
    python $2.py --json
else
    echo "Expected first argument to be either <streamlit> or <cli>"
fi