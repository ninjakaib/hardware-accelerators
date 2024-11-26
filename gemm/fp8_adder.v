module adderFP8 #(parameter FP8_TYPE = 1)(
    input wire [7:0] A,
    input wire [7:0] B,
    input wire clk,
    output reg [7:0] C
);
/*  
    Input:
        A[7:0] -> E4M3 Float
        B[7:0] -> E4M3 Float
    Output:
        C[7:0] -> E4M3 Float (sum of A and B inputs)
        
    Compute the sum of two E4M3 FP8 numbers, A and B.
    No support for Nans or Infs.
    Rounding is towards +Inf or -Inf as necessary
    Design is purely combinational i.e no delay to result
*/

wire [3:0] expA, expB;
wire [2:0] _mantA, _mantB; 
wire signA, signB;

wire redOrExpA,  redOrExpB;

assign redOrExpA = |(expA);
assign redOrExpB = |(expB); 

assign {signA, expA, _mantA} = A;
assign {signB, expB, _mantB} = B;


wire [3:0] mantA, mantB;

assign mantA = {redOrExpA, _mantA};
assign mantB = {redOrExpB, _mantB};

/*
 The arg1, arg2 notation is for internal manipulations
*/

reg sign_diff;
reg [3:0]  exp_diff;

reg [7:0] mant1, mant2;
reg [8:0] mant2S;
reg deg_check; 
reg [8:0] _mant_sum, mant_sum; 
reg [3:0] exp1;

reg [3:0] exp_sum; 

reg is_roundable;
// reg is_degen;

reg [3:0] expAReg, expBReg; 

reg result_sign;
reg gt;

reg [3:0] expSubArg1, expSubArg2;
always @(*) begin
    /*
    Important to note that in E4M3, exponents of 0000 and 0001
    are technically the same. They both represent and exponent of -7
    with `0000` merely indicating that the significand is subnormal.
    
    As such, it is beneficial for futher processing to 'normalize' the exponent.
    We do this here by simply ORing the negation of the reduction OR operation on
    the exponent
    */
    expAReg = expA | (!redOrExpA);
    expBReg = expB | (!redOrExpB); 
    sign_diff = signA ^ signB;
    mant1[3:0] = 4'b0;
    mant2[3:0] = 4'b0;
    gt = {expA, mantA} >= {expB, mantB};
    expSubArg1 = gt? expAReg: expBReg;
    expSubArg2 = gt? expBReg: expAReg;
    exp_diff = expSubArg1 - expSubArg2;
    
    mant1[7:4] = gt? mantA: mantB;
    mant2[7:4] = gt? mantB: mantA;
    result_sign = gt? signA: signB;
    exp1 = gt? expAReg: expBReg;
    
    /*
    This design doesn't use a full-width significand shift/extension.
    
    My intuition was that big shift values would ultimately not affect 
    results significantly, so I thought to take advantage of that to keep
    the mantissa widths to 8 bits

    And it mostly works, save for the case where a a mantissa of x001 is
    shifted right by 5 places.

    The deg_check signal detects that case and remedies it by masquerading
    x001 values as x010 values.

    As this issue only really arises with subtraction, it is handled for that case.
    
    See the next always block 

_       mant_sum = mant1 + (sign_diff? (-(mant2S|deg_check)): mant2S);    
                                              ^^^^^
*/

    {mant2S, deg_check} = {mant2, 1'b0} >> exp_diff; 
end
/* 
{8765} is rounded when bit 4 is set
{7654} is rounded when bit 3 is set
in both cases, adding 1 to bit 4 accomplishes rounding

{6543} is rounded when bit 2 is set
{5432} is rounded when bit 1 is set
in both cases, adding 1 to bit 2 accomplishes rounding

*/

reg [1:0] round; 
always @(*) begin
    _mant_sum = mant1 + (sign_diff? (-(mant2S|deg_check)): mant2S); 
    
    // rounding value generation
    round[1] = (_mant_sum[8] & _mant_sum[4]) || (_mant_sum[7] & _mant_sum[3]);
    round[0] =  !round[1] && ((_mant_sum[6] & _mant_sum[2]) || ( (_mant_sum[5]) && _mant_sum[1])); 

    // rounding value
    mant_sum = {_mant_sum[8:2] + {round[1], 1'b0, round[0]}, _mant_sum[1:0]}; 

end

reg [1:0] exp_neg;// computer number of places to shift left
reg [2:0] sh_req; // number of places to shift left. See comment at signal assignment

reg left_shift; // Attempt left shift

always @(*) begin
    left_shift = !(mant_sum[8] || mant_sum[7]); 
    exp_neg[1] = (left_shift && !mant_sum[6]) && (mant_sum[5] || mant_sum[4]);
    exp_neg[0] = left_shift && (!mant_sum[5] && mant_sum[4] || mant_sum[6]);
end


reg [3:0] final_exp;

reg [3:0] true_shift_or_exp; // 
reg [4:0] exp_sum_arg; // value to add to exponent
reg over_under_flow; // overflow or underflow


always @(*) begin
    /*
    There is a special case where mant1 = b1000_0000 and mant2S = b0111_1000.
    The resulting subtration is b0000_1000.
    As it is the only case where a left shift by 4 is required, we detect it here
    and set the sh_req as ncessary (exp_neg is guaranteed to be b00 when this occurs, so it's alright)
    */
    sh_req = {mant_sum==5'b1000, exp_neg};

    // The logic on the next line inverts (by means of the XOR gate), the value to be shifted
    // as if to find it's 2's complement (but stopping short of adding '1'). As it turns out,
    // this \cancels out' in subsequent computations, while makeing the detection of overflows a
    // lot simpler.
    // The XOR clears sh_req when no left shift is necessary and OR mant_sum[8] makes exp_sum_arg '1'
    // for the case where the mantissa sum overflows to the 8th bit

    exp_sum_arg = ({2'b00, sh_req} ^ {5{left_shift}}) | mant_sum[8]; 
    {over_under_flow, exp_sum} = exp1 + exp_sum_arg;

    // When underflow occurs, the exponent has to be set to 0000, and we need to
    // readjust the left shift as necessary.
    // Note that when underflow does not occur, the left shift amount will always be sh_req
    // when underflow does occur, we must use the value of (exp_sum + 1) as the shift value.
    // The '+1' compensates for the incomplete two's complement on sh_req. This allows us to
    // compute the exponent without introducing any new adder!
    true_shift_or_exp = exp_sum + (over_under_flow?sh_req:1'b1);
end

reg[3:0] final_mant;
reg[8:0] shifted_mant;


always @(*) begin
    // shift by true_shift_or exp when underflow occurs, sh_req when no underflow
    shifted_mant = mant_sum << (over_under_flow? true_shift_or_exp: sh_req);

    if (shifted_mant[8])
        // mantissa should be set to b1111 when overflow occurs (max_value) 
        final_mant = mant_sum[8:5] | {4{over_under_flow}}; 
    else
        final_mant = shifted_mant[7:4];

    if (left_shift) begin
        // exponent set to 0 when underflow occurs, true_shift_or_exp otherwise
        final_exp = {4{!over_under_flow}} & true_shift_or_exp;
    end else begin
        // exponent set to b1111 when overflow occurs
        final_exp = {4{over_under_flow}} | exp_sum[3:0];
    end
    
end


always @(*) begin
    C = {result_sign, final_exp, final_mant[2:0]}; 
end

    
endmodule