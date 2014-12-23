target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() readnone nounwind
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() readnone nounwind
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() readnone nounwind
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() readnone nounwind

define i32 @pos_start_impl() {
  %threadIdx = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %blockIdx = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %blockDim = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %1 = mul nsw i32 %blockIdx, %blockDim
  %2 = add nsw i32 %threadIdx, %1
  ret i32 %2
}

define i32 @pos_step_impl() {
  %blockDim = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %gridDim = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %1 = mul nsw i32 %blockDim, %gridDim
  ret i32 %1
}

; Function Attrs: uwtable
define void @kernel(i8** %byte_stream, i32* nocapture readonly %row_count_ptr, i64* nocapture readonly %agg_init_val, i32* nocapture %out) #3 {
  %1 = getelementptr i8** %byte_stream, i32 0
  %2 = load i8** %1
  %3 = load i32* %row_count_ptr, align 4
  %4 = load i64* %agg_init_val, align 8
  %5 = call i32 @pos_start_impl()
  %6 = call i32 @pos_step_impl()
  %7 = icmp slt i32 %5, %3
  br i1 %7, label %.lr.ph.preheader, label %19

.lr.ph.preheader:                                 ; preds = %0
  br label %.lr.ph

.lr.ph:                                           ; preds = %16, %.lr.ph.preheader
  %result.0 = phi i64 [ %4, %.lr.ph.preheader ], [ %result.1, %16 ]
  %pos.01 = phi i32 [ %17, %16 ], [ %5, %.lr.ph.preheader ]
  %8 = sext i32 %pos.01 to i64
  %9 = getelementptr inbounds i8* %2, i64 %8
  %10 = load i8* %9, align 1
  %11 = sext i8 %10 to i64
  %12 = icmp sgt i64 %11, 41
  %13 = icmp eq i1 %12, 0
  br i1 %13, label %16, label %14

; <label>:14                                      ; preds = %.lr.ph
  %15 = add nsw i64 %result.0, 1
  br label %16

; <label>:16                                      ; preds = %14, %.lr.ph
  %result.1 = phi i64 [ %result.0, %.lr.ph ], [ %15, %14 ]
  %17 = add nsw i32 %pos.01, %6
  %18 = icmp slt i32 %17, %3
  br i1 %18, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %16
  br label %19

; <label>:19                                      ; preds = %._crit_edge, %0
  %20 = phi i64 [ %result.1, %._crit_edge ], [ %4, %0 ]
  %21 = trunc i64 %20 to i32
  %22 = sext i32 %5 to i64
  %23 = getelementptr inbounds i32* %out, i64 %22
  store i32 %21, i32* %23, align 4
  ret void
}

!nvvm.annotations = !{!0}
!0 = metadata !{void (i8**,
                      i32*,
                      i64*,
                      i32*)* @kernel, metadata !"kernel", i32 1}
