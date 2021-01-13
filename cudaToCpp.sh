for f in *.cu; do mv $f `echo $f|sed 's/.cu$/.cpp/1' `;done
