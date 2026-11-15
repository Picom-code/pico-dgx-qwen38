#!/bin/bash
# dist.sh NAME BUILDBOX : copy build NAME from BUILDBOX:~/q38/build/out to ~/q38/tfbin on all three workers; push kit + harness.
# HEAD=user@host of the cluster head must be set (ssh key via your ssh config).
N=$1; BB=$2; 
scp -q kit/*.py kit/*.sh $HEAD:q38/kitpush/ 2>/dev/null || { ssh -n $HEAD 'mkdir -p ~/q38/kitpush'; scp -q kit/*.py kit/*.sh $HEAD:q38/kitpush/; }
scp -q harness/w_*.sh $HEAD:q38/
ssh -n $HEAD "for h in pico@spark-3d24 pico@spark-4e5c picom@spark-923b; do scp -q ~/q38/kitpush/* \$h:q38/kit/; [ -n '$N' ] && { [ \$h = $BB ] && ssh -n \$h 'rm -rf ~/q38/tfbin/$N; cp -a ~/q38/build/out/$N ~/q38/tfbin/$N' || ssh -n $BB 'tar cf - -C ~/q38/build/out $N' | ssh \$h 'rm -rf ~/q38/tfbin/$N; tar xf - -C ~/q38/tfbin'; ssh -n \$h 'sha256sum ~/q38/tfbin/$N/native/bin/tensorfold-native | cut -c1-16; strings ~/q38/tfbin/$N/native/bin/tensorfold-native | grep -c \"${3:-Q38}\"'; }; done"
