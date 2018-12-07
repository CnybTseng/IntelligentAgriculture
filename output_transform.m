function [o0 o1 o2 o3 o4 o5] = output_transform(m0,m1,m2,m3,m4,m5,m6,m7)

m1_add_m2 = m1 + m2;
m1_sub_m2 = m1 - m2;
m3_add_m4 = m3 + m4;
m3_sub_m4 = m3 - m4;
m5_add_m6 = m5 + m6;
m5_sub_m6 = m5 - m6;

s0 = m0 + m1_add_m2;
s5 = m7 + m1_sub_m2;

s1 = m1_sub_m2 + m5_sub_m6 * 16;
s4 = m1_add_m2 + m3_add_m4 * 16;
s2 = m1_add_m2 + m5_add_m6 * 8;
s3 = m1_sub_m2 + m3_sub_m4 * 8;

s0 = s0 + m5_add_m6 * 32;
s5 = s5 + m3_sub_m4 * 32;
s1 = s1 + m3_sub_m4 * 2;
s4 = s4 + m5_add_m6 * 2;

s0 = s0 + m3_add_m4;
s5 = s5 + m5_sub_m6;

s2 = s2 + m3_add_m4 * 4;
s3 = s3 + m5_sub_m6 * 4;

o0 = s0;
o1 = s1;
o2 = s2;
o3 = s3;
o4 = s4;
o5 = s5;