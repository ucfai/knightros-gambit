# example use:
# ./train.sh streamlit
# ./train.sh json

if [ "$1" = "streamlit" ]; then
    streamlit run train.py -- -d
elif [ "$1" = "json" ]; then
    python train.py -j "${@:2}"
else
    echo "Expected first argument to be either <streamlit> or <json>"
fi
