// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, exp_sum, mant_product, out3643, out3771);
    input clk;
    input rst;
    input[4:0] exp_sum;
    input[7:0] mant_product;
    output[3:0] out3643;
    output[3:0] out3771;

    wire[4:0] _ver_out_tmp_0;
    wire const_689_0;
    wire const_690_0;
    wire[1:0] const_691_2;
    wire const_692_1;
    wire const_693_0;
    wire[1:0] const_694_1;
    wire[1:0] const_695_0;
    wire const_696_0;
    wire const_697_0;
    wire const_698_0;
    wire const_699_0;
    wire[1:0] const_700_2;
    wire const_701_1;
    wire const_702_0;
    wire[1:0] const_703_1;
    wire[1:0] const_704_0;
    wire const_705_0;
    wire const_706_0;
    wire const_707_0;
    wire const_708_0;
    wire[1:0] const_709_2;
    wire const_710_1;
    wire const_711_0;
    wire[1:0] const_712_1;
    wire[1:0] const_713_0;
    wire const_714_0;
    wire const_715_0;
    wire const_716_0;
    wire const_717_0;
    wire[1:0] const_718_2;
    wire const_719_1;
    wire const_720_0;
    wire[1:0] const_721_1;
    wire[1:0] const_722_0;
    wire const_723_0;
    wire const_724_0;
    wire[2:0] const_725_4;
    wire[1:0] const_726_1;
    wire const_727_0;
    wire const_728_0;
    wire const_729_0;
    wire[2:0] const_730_4;
    wire[1:0] const_731_1;
    wire const_732_0;
    wire const_733_0;
    wire const_734_0;
    wire[3:0] const_735_8;
    wire[1:0] const_736_1;
    wire const_737_0;
    wire const_738_0;
    wire const_739_0;
    wire const_740_0;
    wire const_742_0;
    wire const_743_0;
    wire[3:0] tmp3643;
    wire[1:0] tmp3644;
    wire[1:0] tmp3645;
    wire[1:0] tmp3646;
    wire[1:0] tmp3647;
    wire[1:0] tmp3648;
    wire tmp3649;
    wire[1:0] tmp3650;
    wire tmp3651;
    wire tmp3652;
    wire[1:0] tmp3653;
    wire tmp3654;
    wire tmp3655;
    wire tmp3656;
    wire tmp3657;
    wire tmp3658;
    wire tmp3659;
    wire tmp3660;
    wire[1:0] tmp3661;
    wire[1:0] tmp3662;
    wire[1:0] tmp3663;
    wire[1:0] tmp3664;
    wire[1:0] tmp3665;
    wire tmp3666;
    wire[1:0] tmp3667;
    wire tmp3668;
    wire tmp3669;
    wire[1:0] tmp3670;
    wire tmp3671;
    wire tmp3672;
    wire tmp3673;
    wire tmp3674;
    wire tmp3675;
    wire tmp3676;
    wire tmp3677;
    wire[1:0] tmp3678;
    wire[1:0] tmp3679;
    wire[1:0] tmp3680;
    wire[1:0] tmp3681;
    wire[1:0] tmp3682;
    wire tmp3683;
    wire[1:0] tmp3684;
    wire tmp3685;
    wire tmp3686;
    wire[1:0] tmp3687;
    wire tmp3688;
    wire tmp3689;
    wire tmp3690;
    wire tmp3691;
    wire tmp3692;
    wire tmp3693;
    wire tmp3694;
    wire[1:0] tmp3695;
    wire[1:0] tmp3696;
    wire[1:0] tmp3697;
    wire[1:0] tmp3698;
    wire[1:0] tmp3699;
    wire tmp3700;
    wire[1:0] tmp3701;
    wire tmp3702;
    wire tmp3703;
    wire[1:0] tmp3704;
    wire tmp3705;
    wire tmp3706;
    wire tmp3707;
    wire tmp3708;
    wire tmp3709;
    wire tmp3710;
    wire tmp3711;
    wire[1:0] tmp3712;
    wire[1:0] tmp3713;
    wire[1:0] tmp3714;
    wire[1:0] tmp3715;
    wire[2:0] tmp3716;
    wire tmp3717;
    wire tmp3718;
    wire tmp3719;
    wire tmp3720;
    wire[2:0] tmp3721;
    wire tmp3722;
    wire tmp3723;
    wire[2:0] tmp3724;
    wire tmp3725;
    wire tmp3726;
    wire tmp3727;
    wire[1:0] tmp3728;
    wire[2:0] tmp3729;
    wire[2:0] tmp3730;
    wire[2:0] tmp3731;
    wire[2:0] tmp3732;
    wire[2:0] tmp3733;
    wire tmp3734;
    wire tmp3735;
    wire tmp3736;
    wire tmp3737;
    wire[2:0] tmp3738;
    wire tmp3739;
    wire tmp3740;
    wire[2:0] tmp3741;
    wire tmp3742;
    wire tmp3743;
    wire tmp3744;
    wire[1:0] tmp3745;
    wire[2:0] tmp3746;
    wire[2:0] tmp3747;
    wire[2:0] tmp3748;
    wire[2:0] tmp3749;
    wire[3:0] tmp3750;
    wire tmp3751;
    wire tmp3752;
    wire tmp3753;
    wire[1:0] tmp3754;
    wire[3:0] tmp3755;
    wire tmp3756;
    wire tmp3757;
    wire[3:0] tmp3758;
    wire tmp3759;
    wire tmp3760;
    wire tmp3761;
    wire[2:0] tmp3762;
    wire[3:0] tmp3763;
    wire[3:0] tmp3764;
    wire[3:0] tmp3765;
    wire[3:0] tmp3766;
    wire[7:0] tmp3767;
    wire[3:0] tmp3768;
    wire[7:0] tmp3769;
    wire[3:0] tmp3770;
    wire[3:0] tmp3771;
    wire[4:0] tmp3772;
    wire tmp3773;
    wire tmp3774;
    wire tmp3775;
    wire tmp3776;
    wire tmp3777;
    wire[4:0] tmp3778;
    wire tmp3779;
    wire tmp3780;
    wire tmp3781;
    wire tmp3782;
    wire tmp3783;
    wire tmp3784;
    wire tmp3785;
    wire tmp3786;
    wire tmp3787;
    wire tmp3788;
    wire tmp3789;
    wire tmp3790;
    wire tmp3791;
    wire tmp3792;
    wire tmp3793;
    wire tmp3794;
    wire tmp3795;
    wire tmp3796;
    wire tmp3797;
    wire tmp3798;
    wire tmp3799;
    wire tmp3800;
    wire tmp3801;
    wire tmp3802;
    wire tmp3803;
    wire[5:0] tmp3804;
    wire tmp3805;
    wire[5:0] tmp3806;
    wire[5:0] tmp3807;
    wire[3:0] tmp3808;

    // Combinational
    assign _ver_out_tmp_0 = 25;
    assign const_689_0 = 0;
    assign const_690_0 = 0;
    assign const_691_2 = 2;
    assign const_692_1 = 1;
    assign const_693_0 = 0;
    assign const_694_1 = 1;
    assign const_695_0 = 0;
    assign const_696_0 = 0;
    assign const_697_0 = 0;
    assign const_698_0 = 0;
    assign const_699_0 = 0;
    assign const_700_2 = 2;
    assign const_701_1 = 1;
    assign const_702_0 = 0;
    assign const_703_1 = 1;
    assign const_704_0 = 0;
    assign const_705_0 = 0;
    assign const_706_0 = 0;
    assign const_707_0 = 0;
    assign const_708_0 = 0;
    assign const_709_2 = 2;
    assign const_710_1 = 1;
    assign const_711_0 = 0;
    assign const_712_1 = 1;
    assign const_713_0 = 0;
    assign const_714_0 = 0;
    assign const_715_0 = 0;
    assign const_716_0 = 0;
    assign const_717_0 = 0;
    assign const_718_2 = 2;
    assign const_719_1 = 1;
    assign const_720_0 = 0;
    assign const_721_1 = 1;
    assign const_722_0 = 0;
    assign const_723_0 = 0;
    assign const_724_0 = 0;
    assign const_725_4 = 4;
    assign const_726_1 = 1;
    assign const_727_0 = 0;
    assign const_728_0 = 0;
    assign const_729_0 = 0;
    assign const_730_4 = 4;
    assign const_731_1 = 1;
    assign const_732_0 = 0;
    assign const_733_0 = 0;
    assign const_734_0 = 0;
    assign const_735_8 = 8;
    assign const_736_1 = 1;
    assign const_737_0 = 0;
    assign const_738_0 = 0;
    assign const_739_0 = 0;
    assign const_740_0 = 0;
    assign const_742_0 = 0;
    assign const_743_0 = 0;
    assign out3643 = tmp3643;
    assign out3771 = tmp3771;
    assign tmp3643 = tmp3770;
    assign tmp3644 = {mant_product[7], mant_product[6]};
    assign tmp3645 = {mant_product[5], mant_product[4]};
    assign tmp3646 = {mant_product[3], mant_product[2]};
    assign tmp3647 = {mant_product[1], mant_product[0]};
    assign tmp3648 = tmp3664;
    assign tmp3649 = {const_690_0};
    assign tmp3650 = {tmp3649, const_689_0};
    assign tmp3651 = tmp3644 == tmp3650;
    assign tmp3652 = {const_693_0};
    assign tmp3653 = {tmp3652, const_692_1};
    assign tmp3654 = tmp3644 == tmp3653;
    assign tmp3655 = ~tmp3651;
    assign tmp3656 = tmp3655 & tmp3654;
    assign tmp3657 = ~tmp3651;
    assign tmp3658 = ~tmp3654;
    assign tmp3659 = tmp3657 & tmp3658;
    assign tmp3660 = {const_697_0};
    assign tmp3661 = {tmp3660, const_696_0};
    assign tmp3662 = tmp3651 ? const_691_2 : tmp3661;
    assign tmp3663 = tmp3656 ? const_694_1 : tmp3662;
    assign tmp3664 = tmp3659 ? const_695_0 : tmp3663;
    assign tmp3665 = tmp3681;
    assign tmp3666 = {const_699_0};
    assign tmp3667 = {tmp3666, const_698_0};
    assign tmp3668 = tmp3645 == tmp3667;
    assign tmp3669 = {const_702_0};
    assign tmp3670 = {tmp3669, const_701_1};
    assign tmp3671 = tmp3645 == tmp3670;
    assign tmp3672 = ~tmp3668;
    assign tmp3673 = tmp3672 & tmp3671;
    assign tmp3674 = ~tmp3668;
    assign tmp3675 = ~tmp3671;
    assign tmp3676 = tmp3674 & tmp3675;
    assign tmp3677 = {const_706_0};
    assign tmp3678 = {tmp3677, const_705_0};
    assign tmp3679 = tmp3668 ? const_700_2 : tmp3678;
    assign tmp3680 = tmp3673 ? const_703_1 : tmp3679;
    assign tmp3681 = tmp3676 ? const_704_0 : tmp3680;
    assign tmp3682 = tmp3698;
    assign tmp3683 = {const_708_0};
    assign tmp3684 = {tmp3683, const_707_0};
    assign tmp3685 = tmp3646 == tmp3684;
    assign tmp3686 = {const_711_0};
    assign tmp3687 = {tmp3686, const_710_1};
    assign tmp3688 = tmp3646 == tmp3687;
    assign tmp3689 = ~tmp3685;
    assign tmp3690 = tmp3689 & tmp3688;
    assign tmp3691 = ~tmp3685;
    assign tmp3692 = ~tmp3688;
    assign tmp3693 = tmp3691 & tmp3692;
    assign tmp3694 = {const_715_0};
    assign tmp3695 = {tmp3694, const_714_0};
    assign tmp3696 = tmp3685 ? const_709_2 : tmp3695;
    assign tmp3697 = tmp3690 ? const_712_1 : tmp3696;
    assign tmp3698 = tmp3693 ? const_713_0 : tmp3697;
    assign tmp3699 = tmp3715;
    assign tmp3700 = {const_717_0};
    assign tmp3701 = {tmp3700, const_716_0};
    assign tmp3702 = tmp3647 == tmp3701;
    assign tmp3703 = {const_720_0};
    assign tmp3704 = {tmp3703, const_719_1};
    assign tmp3705 = tmp3647 == tmp3704;
    assign tmp3706 = ~tmp3702;
    assign tmp3707 = tmp3706 & tmp3705;
    assign tmp3708 = ~tmp3702;
    assign tmp3709 = ~tmp3705;
    assign tmp3710 = tmp3708 & tmp3709;
    assign tmp3711 = {const_724_0};
    assign tmp3712 = {tmp3711, const_723_0};
    assign tmp3713 = tmp3702 ? const_718_2 : tmp3712;
    assign tmp3714 = tmp3707 ? const_721_1 : tmp3713;
    assign tmp3715 = tmp3710 ? const_722_0 : tmp3714;
    assign tmp3716 = tmp3732;
    assign tmp3717 = {tmp3648[1]};
    assign tmp3718 = {tmp3665[1]};
    assign tmp3719 = tmp3717 & tmp3718;
    assign tmp3720 = {tmp3665[0]};
    assign tmp3721 = {const_726_1, tmp3720};
    assign tmp3722 = ~tmp3719;
    assign tmp3723 = tmp3722 & tmp3717;
    assign tmp3724 = {const_727_0, tmp3648};
    assign tmp3725 = ~tmp3719;
    assign tmp3726 = ~tmp3717;
    assign tmp3727 = tmp3725 & tmp3726;
    assign tmp3728 = {const_729_0, const_729_0};
    assign tmp3729 = {tmp3728, const_728_0};
    assign tmp3730 = tmp3719 ? const_725_4 : tmp3729;
    assign tmp3731 = tmp3723 ? tmp3721 : tmp3730;
    assign tmp3732 = tmp3727 ? tmp3724 : tmp3731;
    assign tmp3733 = tmp3749;
    assign tmp3734 = {tmp3682[1]};
    assign tmp3735 = {tmp3699[1]};
    assign tmp3736 = tmp3734 & tmp3735;
    assign tmp3737 = {tmp3699[0]};
    assign tmp3738 = {const_731_1, tmp3737};
    assign tmp3739 = ~tmp3736;
    assign tmp3740 = tmp3739 & tmp3734;
    assign tmp3741 = {const_732_0, tmp3682};
    assign tmp3742 = ~tmp3736;
    assign tmp3743 = ~tmp3734;
    assign tmp3744 = tmp3742 & tmp3743;
    assign tmp3745 = {const_734_0, const_734_0};
    assign tmp3746 = {tmp3745, const_733_0};
    assign tmp3747 = tmp3736 ? const_730_4 : tmp3746;
    assign tmp3748 = tmp3740 ? tmp3738 : tmp3747;
    assign tmp3749 = tmp3744 ? tmp3741 : tmp3748;
    assign tmp3750 = tmp3766;
    assign tmp3751 = {tmp3716[2]};
    assign tmp3752 = {tmp3733[2]};
    assign tmp3753 = tmp3751 & tmp3752;
    assign tmp3754 = {tmp3733[1], tmp3733[0]};
    assign tmp3755 = {const_736_1, tmp3754};
    assign tmp3756 = ~tmp3753;
    assign tmp3757 = tmp3756 & tmp3751;
    assign tmp3758 = {const_737_0, tmp3716};
    assign tmp3759 = ~tmp3753;
    assign tmp3760 = ~tmp3751;
    assign tmp3761 = tmp3759 & tmp3760;
    assign tmp3762 = {const_739_0, const_739_0, const_739_0};
    assign tmp3763 = {tmp3762, const_738_0};
    assign tmp3764 = tmp3753 ? const_735_8 : tmp3763;
    assign tmp3765 = tmp3757 ? tmp3755 : tmp3764;
    assign tmp3766 = tmp3761 ? tmp3758 : tmp3765;
    assign tmp3767 = tmp3769;
    assign tmp3768 = {const_740_0, const_740_0, const_740_0, const_740_0};
    assign tmp3769 = {tmp3768, tmp3750};
    assign tmp3770 = {tmp3767[3], tmp3767[2], tmp3767[1], tmp3767[0]};
    assign tmp3771 = tmp3808;
    assign tmp3772 = exp_sum ^ _ver_out_tmp_0;
    assign tmp3773 = {tmp3772[0]};
    assign tmp3774 = {tmp3772[1]};
    assign tmp3775 = {tmp3772[2]};
    assign tmp3776 = {tmp3772[3]};
    assign tmp3777 = {tmp3772[4]};
    assign tmp3778 = exp_sum & _ver_out_tmp_0;
    assign tmp3779 = {tmp3778[0]};
    assign tmp3780 = {tmp3778[1]};
    assign tmp3781 = {tmp3778[2]};
    assign tmp3782 = {tmp3778[3]};
    assign tmp3783 = {tmp3778[4]};
    assign tmp3784 = tmp3777 & tmp3782;
    assign tmp3785 = tmp3783 | tmp3784;
    assign tmp3786 = tmp3777 & tmp3776;
    assign tmp3787 = tmp3776 & tmp3781;
    assign tmp3788 = tmp3782 | tmp3787;
    assign tmp3789 = tmp3776 & tmp3775;
    assign tmp3790 = tmp3775 & tmp3780;
    assign tmp3791 = tmp3781 | tmp3790;
    assign tmp3792 = tmp3775 & tmp3774;
    assign tmp3793 = tmp3774 & tmp3779;
    assign tmp3794 = tmp3780 | tmp3793;
    assign tmp3795 = tmp3786 & tmp3791;
    assign tmp3796 = tmp3785 | tmp3795;
    assign tmp3797 = tmp3786 & tmp3792;
    assign tmp3798 = tmp3789 & tmp3794;
    assign tmp3799 = tmp3788 | tmp3798;
    assign tmp3800 = tmp3792 & tmp3779;
    assign tmp3801 = tmp3791 | tmp3800;
    assign tmp3802 = tmp3797 & tmp3779;
    assign tmp3803 = tmp3796 | tmp3802;
    assign tmp3804 = {tmp3803, tmp3799, tmp3801, tmp3794, tmp3779, const_742_0};
    assign tmp3805 = {const_743_0};
    assign tmp3806 = {tmp3805, tmp3772};
    assign tmp3807 = tmp3804 ^ tmp3806;
    assign tmp3808 = {tmp3807[3], tmp3807[2], tmp3807[1], tmp3807[0]};

endmodule

