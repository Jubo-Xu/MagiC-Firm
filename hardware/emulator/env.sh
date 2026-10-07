# Source this before configuring/building/running the emulator:
#   source env.sh
# Points cmake at the locally-built SystemC 3.0.1 and puts its shared lib on the
# runtime path.
export SystemCLanguage_DIR="$HOME/opt/systemc-3.0.1/lib/cmake/SystemCLanguage"
export LD_LIBRARY_PATH="$HOME/opt/systemc-3.0.1/lib:${LD_LIBRARY_PATH:-}"
