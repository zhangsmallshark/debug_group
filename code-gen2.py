import os
import sys
import subprocess
import codecs
import math
import json


def extract_blocks(json_file):
    inputs = [[128, 128, 28, 28]]    
    all_blocks = []
    with open(json_file) as j_f:
        mask = json.load(j_f)
        # for k0, v0 in mask.items():
        v0 = mask['block3.layer.2.conv1.weight']
        filter_size = v0['size']
        layer_blocks = v0['block']

        filters_ptr = [0]
        filters = []
        channels_ptr = [0]
        channels = []
        for b in layer_blocks:
            filters_ptr.append(filters_ptr[-1]+len(b[0]))
            filters.extend(b[0])
            channels_ptr.append(channels_ptr[-1]+len(b[1]))
            channels.extend(b[1])

        filters_ptr = [str(i) for i in filters_ptr]
        filters = [str(i) for i in filters]
        channels_ptr = [str(i) for i in channels_ptr]
        channels = [str(i) for i in channels]

        all_blocks.append([filters_ptr, filters, channels_ptr, channels])

    return inputs, all_blocks


def build_switch(H, W):
    R = 3
    S = 3
    Ho = H - (R-1)
    Wo = W - (S-1)
    template = '\t\tcase {}:\n\t\t\tfor ( int r = {}; r < {}; r++) {{\n\t\t\t\tfor ( int s = {}; s < {}; s++) {{' \
               '\n\t\t\t\t\tfloat result = v * temp_kernel[r*S+s];\n\t\t\t\t\ttemp_result[({}-r)*{}+({}-s)] += result;\n\t\t\t\t}}\n\t\t\t}}\n\t\tbreak;\n'
    line = '__device__ void switch_function(int switch_condition,float *temp_kernel,float v,float *temp_result){\n' \
           '\tswitch (switch_condition) {\n'
    for h in range(H):
        for w in range(W):
            r_end = R
            s_end = S
            id = h*W+w
            r_start_condition = (h - Ho + 1)
            r_end_condition = (h+1)
            s_start_condition = (w - Wo + 1)
            s_end_condition = (w+1)
            r_end = min(r_end,r_end_condition)
            r_start = max(0,r_start_condition)
            s_end = min(s_end,s_end_condition)
            s_start = max(0,s_start_condition)
            case_line = template.format(id,r_start,r_end,s_start,s_end,h,(Wo),w)
            line +=case_line
    line += '\n\t}'
    line += '\n}'
    return line


if __name__ == '__main__':

    inputs, all_blocks = extract_blocks('./mask.json')

    max_shared_mem_size = 40000
    TBS = [1,2,4,8,16,32]
    THS = [1,2,3,4,5,6,7]
    ths = [2,3,4,5,6,7]
    tws = [2,3,4,5,6,7]
    tcs = [1,2,4,8]
    exc_file = './group-conv'

    # inputs = [(64,32,224,224),(64,32,112,112),(32,32,56,56),(64,32,56,56),(64,64,56,56),(32,32,28,28),(64,32,28,28)
    #     ,(96,64,28,28),(160,96,28,28),(192,96,28,28),(32,32,14,14),(64,32,14,14),(128,96,14,14),(192,96,14,14),(32,32,7,7),
    #           (64,32,7,7),(96,64,7,7),(192,160,7,7)]

    reader = codecs.open('group-conv-template.cu', 'r', 'utf-8')
    temp_lines = reader.readlines()
    reader.close()

    left_brace = '{'
    right_brace = '}'
    i_idx = 0
    for input_size in inputs:
        C = input_size[0]
        N = input_size[1]
        H = input_size[2]
        W = input_size[3]

        layer_blocks = all_blocks[i_idx]
        i_idx += 1

        for TB in TBS:
            for TH in THS:
                for th in ths:
                    if th > TH:
                        continue
                    for tw in tws:
                        for tc in tcs:
                            if (TH+2)*(W+2)*tc >= max_shared_mem_size:
                                continue

                            lines = temp_lines[:]
                            threads = math.ceil(TH/th) * math.ceil(W/tw) * N
                            if threads > 1024:
                                continue
                            out_line = ''
                            for line in lines:
                                out_line += line

                            switch_func_lines = build_switch(th+2, tw+2)
                            out_line = out_line.replace('#define INTERNAL_TH place holder', '#define INTERNAL_TH {}'.format(th))
                            out_line = out_line.replace('#define INTERNAL_TW place holder', '#define INTERNAL_TW {}'.format(tw))
                            out_line = out_line.replace('#define TH place holder', '#define TH {}'.format(TH))
                            out_line = out_line.replace('#define TC place holder', '#define TC {}'.format(tc))
                            out_line = out_line.replace('#define H place holder', '#define H {}'.format(H))
                            out_line = out_line.replace('#define W place holder', '#define W {}'.format(W))
                            out_line = out_line.replace('#define C place holder', '#define C {}'.format(C))
                            out_line = out_line.replace('#define N place holder', '#define N {}'.format(N))
                            out_line = out_line.replace('#define TB place holder', '#define TB {}'.format(TB))
                            out_line = out_line.replace('switch_function_place_holder', switch_func_lines)

                            num_groups = len(layer_blocks[0]) - 1
                            num_b_filters = len(layer_blocks[1])
                            num_b_channels = len(layer_blocks[3])
                            out_line = out_line.replace('#define PTR_S place holder', f'#define PTR_S {num_groups+1}')
                            out_line = out_line.replace('#define C_S place holder', f'#define C_S {num_b_channels}')
                            out_line = out_line.replace('#define F_S place holder', f'#define F_S {num_b_filters}')
                            out_line = out_line.replace('groups_place_holder', f'int groups = {num_groups};')
                            out_line = out_line.replace('filters_ptr_place_holder', f'int filters_ptr[{num_groups+1}] = {left_brace}{", ".join(layer_blocks[0])}{right_brace};')
                            out_line = out_line.replace('filters_place_holder', f'int filters[{num_b_filters}] = {left_brace}{", ".join(layer_blocks[1])}{right_brace};')
                            out_line = out_line.replace('channels_ptr_place_holder', f'int channels_ptr[{num_groups+1}] = {left_brace}{", ".join(layer_blocks[2])}{right_brace};')
                            out_line = out_line.replace('channels_place_holder', f'int channels[{num_b_channels}] = {left_brace}{", ".join(layer_blocks[3])}{right_brace};')
                            writter = codecs.open('group-conv.cu', 'w+', 'utf-8')
                            writter.write(out_line)

                            subprocess.run(["make", "clean"])
                            subprocess.run(["make", "group-conv"])
                            subprocess.run([exc_file])

