if [ "$#" -ne 2 ]; then
    echo  "./psam_to_jaspar_format.sh  <motif.psam> <motif_len>"
    exit
fi

psam_fn=$1
motif_len=$2

head -n $((motif_len + 4)) < $psam_fn| tail -n $motif_len | awk '{print substr($1,3,2)" " substr($2, 3, 2) " " substr($3, 3, 2)" " substr($4, 3, 2)}' |awk '
{ 
    for (i=1; i<=NF; i++)  {
        a[NR,i] = $i
    }
}
NF>p { p = NF }
END {    
    for(j=1; j<=p; j++) {
        str=a[1,j]
        for(i=2; i<=NR; i++){
            str=str" "a[i,j];
        }
        print str
    }
}'

