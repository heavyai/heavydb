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
define void @kernel(i8** %byte_stream, i32* nocapture readonly %row_count_ptr, i32* nocapture %out) #2 {
  %1 = getelementptr i8** %byte_stream, i32 0
  %2 = load i8** %1
  %3 = load i32* %row_count_ptr, align 4
  %4 = call i32 @pos_start_impl()
  %5 = call i32 @pos_step_impl()
  %6 = icmp slt i32 %4, %3
  br i1 %6, label %.lr.ph.preheader, label %15

.lr.ph.preheader:                                 ; preds = %0
  br label %.lr.ph

.lr.ph:                                           ; preds = %.lr.ph, %.lr.ph.preheader
  %pos.02 = phi i32 [ %13, %.lr.ph ], [ %4, %.lr.ph.preheader ]
  %result.01 = phi i64 [ %result.0., %.lr.ph ], [ 0, %.lr.ph.preheader ]
  %7 = sext i32 %pos.02 to i64
  %8 = getelementptr inbounds i8* %2, i64 %7
  %9 = load i8* %8, align 1
  %10 = sext i8 %9 to i64
  %11 = icmp sgt i64 %10, 15
  %not. = icmp ne i1 %11, 0
  %12 = zext i1 %not. to i64
  %result.0. = add nsw i64 %12, %result.01
  %13 = add nsw i32 %pos.02, %5
  %14 = icmp slt i32 %13, %3
  br i1 %14, label %.lr.ph, label %._crit_edge

._crit_edge:                                      ; preds = %.lr.ph
  %result.0..lcssa = phi i64 [ %result.0., %.lr.ph ]
  %phitmp = trunc i64 %result.0..lcssa to i32
  br label %15

; <label>:15                                      ; preds = %._crit_edge, %0
  %result.0.lcssa = phi i32 [ %phitmp, %._crit_edge ], [ 0, %0 ]
  %16 = sext i32 %4 to i64
  %17 = getelementptr inbounds i32* %out, i64 %16
  store i32 %result.0.lcssa, i32* %17, align 4
  ret void
}

!nvvm.annotations = !{!0}
!0 = metadata !{void (i8**,
                      i32*,
                      i32*)* @kernel, metadata !"kernel", i32 1}
