for f in *.cpp; do mv $f `echo $f|sed 's/.cpp$/.cu/1' `;done
