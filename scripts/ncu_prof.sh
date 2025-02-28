N=(400 256 200 128 64)
W=(72 64 56 48)
C=(384 224 192)
cnt=0

echo "" &> temp.log

for n in ${N[@]}; do
	for w in ${W[@]}; do
		for c in ${C[@]}; do
			
			bash ./ncu_prof_cmd.sh $n $w $c temp.log
			cnt=$(($cnt+1))
			echo "-------------------------------------"
			echo "------------ $cnt out of 60 ---------------"
			echo "-------------------------------------"
		done
	done
done
     