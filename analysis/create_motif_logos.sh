#/bin/sh
# script for generating motif logos from PSAMs using the reduce suite LogoGenerator.

if [ "$#" -ne 2 ]; then
    echo  "./create_motif_logos.sh <result_dir> <n_motifs>"
    exit
fi

result_dir=$1
n_motifs=$2

# generate logos for final psams in a new directory
echo "creating final motif logos"
if [ ! -d $result_dir"/final" ]; then
    mkdir $result_dir"/final"
fi
ls $result_dir | grep "final.*psam" | while read l; do
    fn=$result_dir"/"$l
    LogoGenerator -file=$fn -output=$result_dir"/final/"
    done

# generate logos for each of the motifs from each of the saved checkpoint iterations
for i in `seq 1 $n_motifs`; do
    echo "creating logos for motif" $i
    new_dir=`printf "%s/motif%02d" $result_dir $i`
    if [ ! -d $new_dir ]; then
        mkdir $new_dir
    fi
    ls $result_dir |grep `printf "motif%02d_" $i` | while read l; do 
        logo_fn=$new_dir"/"${l:0:${#l}-4}"png"
        if [ ! -f $logo_fn ]; then 
            LogoGenerator -file=$result_dir"/"$l -logo=$logo_fn
        fi
        done
    done
