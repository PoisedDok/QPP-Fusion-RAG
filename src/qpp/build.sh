#!/bin/bash
# Build script for Java QPP Bridge
# Compiles QPPBridge.java with Gson dependency

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIB_DIR="$SCRIPT_DIR/../../lib"
GSON_JAR="$LIB_DIR/gson-2.11.0.jar"

# Create lib directory
mkdir -p "$LIB_DIR"

# Check Gson
if [ ! -f "$GSON_JAR" ]; then
    echo "ERROR: Gson not found at $GSON_JAR"
    echo "Copy from: sim/packages/lucene-msmarco/target/dependency/gson-2.11.0.jar"
    exit 1
fi

# Clean old compiled files
rm -f "$SCRIPT_DIR"/*.class

# Compile QPPBridge.java (it's self-contained, no other classes needed)
echo "Compiling QPPBridge.java..."
javac -cp "$GSON_JAR" -d "$SCRIPT_DIR/.." "$SCRIPT_DIR/QPPBridge.java"

# Test
echo ""
echo "Testing QPP Bridge..."
echo '{"query":"test query","documents":[{"score":0.9},{"score":0.7},{"score":0.5}],"methods":["RSD","NQC"]}' | \
    java -cp "$SCRIPT_DIR/..:$GSON_JAR" qpp.QPPBridge

echo ""
echo "✅ Build successful! QPP Bridge is ready."
echo "   Class files: $SCRIPT_DIR/../qpp/"
echo "   Gson JAR: $GSON_JAR"

