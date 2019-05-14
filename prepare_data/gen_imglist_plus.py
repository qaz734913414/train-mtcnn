import argparse
import numpy as np
import numpy.random as npr
import os,sys
sys.path.append(os.getcwd())
from config import config

def gen_imglist(size=20, base_num=1, with_hard=False):
    with open('%s/prepare_data/%d/part.txt'%(config.root, size), 'r') as f:
        part = f.readlines()
		
    with open('%s/prepare_data/%d/pos.txt'%(config.root, size), 'r') as f:
        pos = f.readlines()

    with open('%s/prepare_data/%d/neg.txt'%(config.root, size), 'r') as f:
        neg = f.readlines()

    if with_hard:
        if size == 24:
            with open('%s/prepare_data/%d/neg_hard1.txt'%(config.root, size), 'r') as f:
                neg_hard1 = f.readlines()
            with open('%s/prepare_data/%d/neg_hard2.txt'%(config.root, size), 'r') as f:
                neg_hard2 = f.readlines()
        elif size == 48:
            with open('%s/prepare_data/%d/neg_hard1.txt'%(config.root, size), 'r') as f:
                neg_hard1 = f.readlines()
            with open('%s/prepare_data/%d/neg_hard2.txt'%(config.root, size), 'r') as f:
                neg_hard2 = f.readlines()
            with open('%s/prepare_data/%d/neg_hard3.txt'%(config.root, size), 'r') as f:
                neg_hard3 = f.readlines()
        else:
            with open('%s/prepare_data/%d/neg_hard.txt'%(config.root, size), 'r') as f:
                neg_hard = f.readlines()
    
    neg_hard_num = 0
    neg_hard1_num = 0
    neg_hard2_num = 0
    neg_hard3_num = 0
    pos_num = base_num*100000 
    part_num = base_num*100000  
    if with_hard:
        if size == 24:
            neg_num = base_num*150000  
            neg_hard1_num = base_num*75000
            neg_hard2_num = base_num*75000
        elif size == 48:
            neg_num = base_num*120000  
            neg_hard1_num = base_num*60000
            neg_hard2_num = base_num*60000
            neg_hard3_num = base_num*60000
        else:
            neg_num = base_num*200000  
            neg_hard_num = base_num*100000 
    else:
        neg_num = base_num*300000
    
    if with_hard:
        out_file = "%s/prepare_data/%d/train_%d_with_hard_%d.txt"%(config.root, size, size, base_num)
    else:
        out_file = "%s/prepare_data/%d/train_%d_%d.txt"%(config.root, size, size, base_num)
    with open(out_file, "w") as f:
        if len(pos) > pos_num:
            pos_keep = npr.choice(len(pos), size=pos_num, replace=False)
            print('pos_num=%d'%pos_num)
            for i in pos_keep:
                f.write(pos[i])			
        else:
            print('pos_num=%d'%len(pos))
            f.writelines(pos)
        if len(part) > part_num:
            part_keep = npr.choice(len(part), size=part_num, replace=False)
            print('part_num=%d'%part_num)
            for i in part_keep:
                f.write(part[i])			
        else:
            print('part_num=%d'%len(part))
            f.writelines(part)
        if len(neg) > neg_num:
            neg_keep = npr.choice(len(neg), size=neg_num, replace=False)
            print('neg_num=%d'%neg_num)
            for i in neg_keep:
                f.write(neg[i])			
        else:
            print('neg_num=%d'%len(neg))
            f.writelines(neg)

        if with_hard:
            if size == 24:
                if len(neg_hard1) > neg_hard1_num:
                    neg_hard1_keep = npr.choice(len(neg_hard1), size=neg_hard1_num, replace=False)
                    print('neg_hard1_num=%d'%neg_hard1_num)
                    for i in neg_hard1_keep:
                        f.write(neg_hard1[i])			
                else:
                    print('neg_hard1_num=%d'%len(neg_hard1))
                    f.writelines(neg_hard1)
                if len(neg_hard2) > neg_hard2_num:
                    neg_hard2_keep = npr.choice(len(neg_hard2), size=neg_hard2_num, replace=False)
                    print('neg_hard2_num=%d'%neg_hard2_num)
                    for i in neg_hard2_keep:
                        f.write(neg_hard2[i])			
                else:
                    print('neg_hard2_num=%d'%len(neg_hard2))
                    f.writelines(neg_hard2)
            elif size == 48:
                if len(neg_hard1) > neg_hard1_num:
                    neg_hard1_keep = npr.choice(len(neg_hard1), size=neg_hard1_num, replace=False)
                    print('neg_hard1_num=%d'%neg_hard1_num)
                    for i in neg_hard1_keep:
                        f.write(neg_hard1[i])			
                else:
                    print('neg_hard1_num=%d'%len(neg_hard1))
                    f.writelines(neg_hard1)
                if len(neg_hard2) > neg_hard2_num:
                    neg_hard2_keep = npr.choice(len(neg_hard2), size=neg_hard2_num, replace=False)
                    print('neg_hard2_num=%d'%neg_hard2_num)
                    for i in neg_hard2_keep:
                        f.write(neg_hard2[i])		
                else:
                    print('neg_hard2_num=%d'%len(neg_hard2))
                    f.writelines(neg_hard2)
                if len(neg_hard3) > neg_hard3_num:
                    neg_hard3_keep = npr.choice(len(neg_hard3), size=neg_hard3_num, replace=False)
                    print('neg_hard3_num=%d'%neg_hard3_num)
                    for i in neg_hard3_keep:
                        f.write(neg_hard3[i])		
                else:
                    print('neg_hard3_num=%d'%len(neg_hard3))
                    f.writelines(neg_hard3)
            else:
                if len(neg_hard) > neg_hard_num:
                    neg_hard_keep = npr.choice(len(neg_hard), size=neg_hard_num, replace=False)
                    print('neg_hard_num=%d'%neg_hard_num)
                    for i in neg_hard_keep:
                        f.write(neg_hard[i])			
                else:
                    print('neg_hard_num=%d'%len(neg_hard))
                    f.writelines(neg_hard)
        
def parse_args():
    parser = argparse.ArgumentParser(description='Train proposal net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--size', dest='size', help='20 or 24 or 48', default='20', type=str)
    parser.add_argument('--base_num', dest='base_num', help='base num', default='1', type=str)
    parser.add_argument('--with_hard', dest='with_hard', help='with_hard', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    gen_imglist(int(args.size), int(args.base_num), args.with_hard)