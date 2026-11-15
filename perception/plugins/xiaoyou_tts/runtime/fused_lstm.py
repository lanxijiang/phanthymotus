#!/usr/bin/env python3
"""Private TRT10.4 FP32/cuDNN plugin for the six pinned original encoder/acoustic LSTMs.

Module import is stdlib-only: no TensorRT import, CDLL, CUDA or NN.
Explicit make_plugin_classes(trt)/register_creator(trt) expose the old class API.
GPU resource creation happens only in explicit TensorRT workspace/shape callbacks.
No production file, ONNX graph, plan, dependency or shared library is changed.
"""
from pathlib import Path
import ast, hashlib, json, re, sys, threading, time, types, weakref
sys.dont_write_bytecode=True
NAME='XiaoyouBiLSTMFP32';VERSION='1'
NAMESPACE='lanxijiang_xiaoyou_fp32_lstm_v2'
C=640;H=256;N_LIMIT=512;CAP_BYTES=512*2**20;WORKSPACE_BUDGET=64*2**20
ABI_SHA='a79ce59f7617bb44f53ef0f4122d2bff1b653ce90f41e73bd85fd244ee1b3761'
ALLOWED_METADATA={'encoder:/prosody/lstms.0/LSTM': {'type': 'XiaoyouBiLSTMFP32', 'version': '1', 'namespace': 'lanxijiang_xiaoyou_fp32_lstm_v2', 'node_id': 'encoder:/prosody/lstms.0/LSTM', 'C': 640, 'H': 256, 'B': 1, 'N_limit': 512, 'precision': 'FP32', 'math': 'CUDNN_FMA_MATH', 'algorithm': 'STANDARD', 'input_layout': '[N,1,C]', 'output_layout': '[N,2,1,256]', 'gate_order': 'IOFC', 'h0_c0': 'zero_NULL', 'sequence_lens': 'full_N_host_descriptor_devSeqLengths_NULL', 'original_weight_sha256': {'W': '4dc89597695a4a1854af96ce735ee3d7943d3de2abd5177802d1761e73120908', 'R': 'b0316424c11681f26d2e002bbe71f3b8f4832328c2e830b8de22c9a363702c61', 'B': 'f96fc22ad70d691f07b04cbe5b3c8627e21ace95478e17009970b0601ef2d839'}, 'weight_shapes': {'W': [2, 1024, 640], 'R': [2, 1024, 256], 'B': [2, 2048]}}, 'encoder:/prosody/lstms.2/LSTM': {'type': 'XiaoyouBiLSTMFP32', 'version': '1', 'namespace': 'lanxijiang_xiaoyou_fp32_lstm_v2', 'node_id': 'encoder:/prosody/lstms.2/LSTM', 'C': 640, 'H': 256, 'B': 1, 'N_limit': 512, 'precision': 'FP32', 'math': 'CUDNN_FMA_MATH', 'algorithm': 'STANDARD', 'input_layout': '[N,1,C]', 'output_layout': '[N,2,1,256]', 'gate_order': 'IOFC', 'h0_c0': 'zero_NULL', 'sequence_lens': 'full_N_host_descriptor_devSeqLengths_NULL', 'original_weight_sha256': {'W': '8412d7352a4fb6aa50237c7f55d95ed5bbc3f8e9c445f0d98c0e52afdaa611b1', 'R': '4f259dd2e007fce89595d0d8ab2a17d441dc988b4c226bdc7b5adcd4937f228b', 'B': 'd17f91b45d66346857a779c3ae1c160c0f8f70acaea7349387810998313cd185'}, 'weight_shapes': {'W': [2, 1024, 640], 'R': [2, 1024, 256], 'B': [2, 2048]}}, 'encoder:/prosody/lstms.4/LSTM': {'type': 'XiaoyouBiLSTMFP32', 'version': '1', 'namespace': 'lanxijiang_xiaoyou_fp32_lstm_v2', 'node_id': 'encoder:/prosody/lstms.4/LSTM', 'C': 640, 'H': 256, 'B': 1, 'N_limit': 512, 'precision': 'FP32', 'math': 'CUDNN_FMA_MATH', 'algorithm': 'STANDARD', 'input_layout': '[N,1,C]', 'output_layout': '[N,2,1,256]', 'gate_order': 'IOFC', 'h0_c0': 'zero_NULL', 'sequence_lens': 'full_N_host_descriptor_devSeqLengths_NULL', 'original_weight_sha256': {'W': 'f4f00ab7e7373470bb8f988c746797e282f5d87543d187cc1e062bb5f4f62800', 'R': 'ee4f983c070ab5253183e12eb02fa533d8b8a76b024991f044cc0fb9e84b17f0', 'B': '1f097376391686187054a739c37ab27001098f31929ad6e87184b70d9096ab1b'}, 'weight_shapes': {'W': [2, 1024, 640], 'R': [2, 1024, 256], 'B': [2, 2048]}}, 'encoder:/lstm/LSTM': {'type': 'XiaoyouBiLSTMFP32', 'version': '1', 'namespace': 'lanxijiang_xiaoyou_fp32_lstm_v2', 'node_id': 'encoder:/lstm/LSTM', 'C': 640, 'H': 256, 'B': 1, 'N_limit': 512, 'precision': 'FP32', 'math': 'CUDNN_FMA_MATH', 'algorithm': 'STANDARD', 'input_layout': '[N,1,C]', 'output_layout': '[N,2,1,256]', 'gate_order': 'IOFC', 'h0_c0': 'zero_NULL', 'sequence_lens': 'full_N_host_descriptor_devSeqLengths_NULL', 'original_weight_sha256': {'W': 'ad196a98bc49edbf8847c4391be572653d96c2b69a84eefc5c73ac8315dbe859', 'R': '34332d1597ce57b348cf939d7ed60283d2538f71bddf38943ede7219745d3eee', 'B': '84fc4d8ac85f40e0841190eb3c8b2cde274b6b3192e8219931df0b420f8ff0c6'}, 'weight_shapes': {'W': [2, 1024, 640], 'R': [2, 1024, 256], 'B': [2, 2048]}}, 'encoder:/text_encoder/lstm/LSTM': {'type': 'XiaoyouBiLSTMFP32', 'version': '1', 'namespace': 'lanxijiang_xiaoyou_fp32_lstm_v2', 'node_id': 'encoder:/text_encoder/lstm/LSTM', 'C': 512, 'H': 256, 'B': 1, 'N_limit': 512, 'precision': 'FP32', 'math': 'CUDNN_FMA_MATH', 'algorithm': 'STANDARD', 'input_layout': '[N,1,C]', 'output_layout': '[N,2,1,256]', 'gate_order': 'IOFC', 'h0_c0': 'zero_NULL', 'sequence_lens': 'full_N_host_descriptor_devSeqLengths_NULL', 'original_weight_sha256': {'W': '6fb6a58743cd381e03d813b04afc69a55b5267f35f3e6f3b4c86e654f99a795e', 'R': '023c76b13b03ab34c1a984b67c68fc7e872088826eb4d8af9afb7a40f6b7400d', 'B': 'ff46a9dca06149ec9c6b4147b824d66a66e5118f555961abe83795b32b9d8d54'}, 'weight_shapes': {'W': [2, 1024, 512], 'R': [2, 1024, 256], 'B': [2, 2048]}}, 'acoustic:/shared/LSTM': {'type': 'XiaoyouBiLSTMFP32', 'version': '1', 'namespace': 'lanxijiang_xiaoyou_fp32_lstm_v2', 'node_id': 'acoustic:/shared/LSTM', 'C': 640, 'H': 256, 'B': 1, 'N_limit': 832, 'precision': 'FP32', 'math': 'CUDNN_FMA_MATH', 'algorithm': 'STANDARD', 'input_layout': '[N,1,C]', 'output_layout': '[N,2,1,256]', 'gate_order': 'IOFC', 'h0_c0': 'zero_NULL', 'sequence_lens': 'full_N_host_descriptor_devSeqLengths_NULL', 'original_weight_sha256': {'W': '61dbfe4e9b36df8518b89eb4fc6dcbb9a74f39d174a3cf6335436e22c39f0522', 'R': '38c8bc9a74b87ad34b76144eec3fe1f53deb1b42bbaf4fe84a4509d15da690ff', 'B': '60d30ff787954c278a6739d3f8eba483091ae25ce1962a186b9ffe34fd8f29a6'}, 'weight_shapes': {'W': [2, 1024, 640], 'R': [2, 1024, 256], 'B': [2, 2048]}}}
SOURCES=[
 'https://raw.githubusercontent.com/NVIDIA/TensorRT/release/10.3/python/src/infer/pyPlugin.cpp',
 'https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/plugins-python.html',
 'https://docs.nvidia.com/deeplearning/cudnn/backend/v9.4.0/api/cudnn-adv-library.html']
_CLASSES={};_REGISTERED={};_LOCK=threading.RLock();_INSTANCES=weakref.WeakValueDictionary()

def need(value,message):
    if not value:raise RuntimeError(message)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def checked_metadata(value):
    need(isinstance(value,dict) and value.get('node_id') in ALLOWED_METADATA,'Unknown original LSTM')
    expected=ALLOWED_METADATA[value['node_id']]
    need(value==expected,'Original FP32 LSTM contract differs')
    return json.loads(json.dumps(expected))

def _probe_definitions(np):
    """Use the reviewed production ABI definitions verbatim."""
    import ctypes as ct
    import platform
    path=Path(__file__).with_name('cudnn_api.py');need(digest(path)==ABI_SHA,'Reviewed cuDNN ABI source changed')
    selected={'need','raw_sha','APIError','API','Resources','tensor_shape','dense_strides','rnn_readback'}
    tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in selected]
    need({n.name for n in nodes}==selected,'Reviewed ABI definition set differs')
    scope={'ct':ct,'np':np,'platform':platform,'hashlib':hashlib,'json':json,'H':H,'CAP_BYTES':CAP_BYTES,
        'P':ct.c_void_p,'I':ct.c_int,'S':ct.c_size_t,'U':ct.c_uint32,
        'PP':ct.POINTER(ct.c_void_p),'PI':ct.POINTER(ct.c_int),'PS':ct.POINTER(ct.c_size_t)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<unchanged_reviewed_cudnn_ABI_definitions>','exec'),scope)
    return types.SimpleNamespace(**scope)

class _Runtime:
    """One plugin instance's private handle/descriptors/weights; never shared by clone."""
    def __init__(self,weights,np,meta):
        self.C=meta['C'];self.N_limit=meta['N_limit']
        started=time.perf_counter()
        self.np=np;self.p=_probe_definitions(np);self.q=None;self.last_stream=None;self.current_N=None
        self.cache={};self.closed=False
        self.report={'prototype':True,'ready':False,'initialization_syncs':0,
            'enqueue_syncs':0,'enqueue_D2H':0,'enqueue_CPU_NN':False,
            'cap_scope':'plugin cudaMalloc + borrowed TRT workspace/input/output, excludes hidden library allocations',
            'workspace_budget_bytes':WORKSPACE_BUDGET,
            'workspace_bound_scope':'Fixed64MiB TRT scratch budget; query actual N and reject excess, no claim all N are proven'}
        p=self.p;ct=p.ct;self.api=p.API()
        try:
            self.q=q=p.Resources(self.api,self.report,None) # owns only this private initialization stream
            self.drop=q.descriptor('Dropout');self.rnn=q.descriptor('RNN')
            states=p.S();self.api.call('cudnnDropoutGetStatesSize',q.handle,ct.byref(states))
            self.drop_states=q.allocate('dropout_states',states.value)
            self.api.call('cudnnSetDropoutDescriptor',self.drop,q.handle,0.0,self.drop_states,states.value,42)
            self.api.call('cudnnSetRNNDescriptor_v8',self.rnn,0,2,2,1,0,0,0,3,self.C,H,H,1,self.drop,0)
            actual,flags=p.rnn_readback(self.api,self.rnn)
            need(list(actual.values())==[0,2,2,1,0,0,0,3,self.C,H,H,1]and flags==0,'Exact STANDARD FP32/FMA descriptor required')
            self.report['actual_RNN_descriptor']=actual
            self.xdesc=q.descriptor('RNNData');self.ydesc=q.descriptor('RNNData');self.query_desc=q.descriptor('RNNData')
            self.hdesc=q.descriptor('Tensor');self.cdesc=q.descriptor('Tensor')
            dims,strides=(p.I*3)(2,1,H),(p.I*3)(H,H,1)
            for descriptor in (self.hdesc,self.cdesc):self.api.call('cudnnSetTensorNdDescriptor',descriptor,0,3,dims,strides)
            weight_bytes=p.S();self.api.call('cudnnGetRNNWeightSpaceSize',q.handle,self.rnn,ct.byref(weight_bytes))
            self.weight_bytes=weight_bytes.value;self.weights=q.allocate('weights',self.weight_bytes)
            self.api.call('cudaMemsetAsync',self.weights,0,self.weight_bytes,q.stream)
            matrix_desc=q.descriptor('Tensor');bias_desc=q.descriptor('Tensor');intervals=[];keep=[]
            # Exactly the successful probe's IOFC -> IFCO matrix/bias mapping.
            for direction in range(2):
                for linear_id in range(8):
                    matrix_ptr,bias_ptr=p.P(),p.P()
                    self.api.call('cudnnGetRNNWeightParams',q.handle,self.rnn,direction,self.weight_bytes,self.weights,linear_id,matrix_desc,ct.byref(matrix_ptr),bias_desc,ct.byref(bias_ptr))
                    gate=(0,2,3,1)[linear_id%4];recurrent=linear_id>=4;width=H if recurrent else self.C;offset=4*H if recurrent else 0
                    matrix=np.ascontiguousarray(weights['R'if recurrent else'W'][direction,gate*H:(gate+1)*H,:])
                    bias=np.ascontiguousarray(weights['B'][direction,offset+gate*H:offset+(gate+1)*H])
                    for ptr,descriptor,array,wanted in [(matrix_ptr,matrix_desc,matrix,[1,H,width]),(bias_ptr,bias_desc,bias,[1,H,1])]:
                        dtype,shape,step=p.tensor_shape(self.api,descriptor)
                        need(dtype==0 and shape==wanted and step==p.dense_strides(shape),'Unexpected cuDNN parameter packing; no guessed transpose')
                        need(ptr.value is not None,'Missing original matrix/bias')
                        begin=ptr.value-self.weights.value;end=begin+array.nbytes
                        need(0<=begin<end<=self.weight_bytes and all(end<=a or begin>=b for a,b in intervals),'Invalid/overlapping weight space')
                        intervals.append((begin,end));keep.append(array)
                        self.api.call('cudaMemcpyAsync',ptr,p.P(array.ctypes.data),array.nbytes,1,q.stream)
            # Sole startup copy barrier, never inside enqueue. No device-to-host copy.
            self.api.call('cudaStreamSynchronize',q.stream);self.report['initialization_syncs']=1
            self.report['parameter_ranges']=len(intervals);self.report['weight_H2D_once']=True
            self.report['one_time_descriptor_and_weight_setup_wall_ms']=(time.perf_counter()-started)*1000
        except BaseException:
            self.close();raise

    def query(self,n):
        need(not self.closed and 1<=n<=self.N_limit,'N out of prototype bound')
        if n not in self.cache:
            p=self.p;seq=(p.I*1)(n);padding=p.ct.c_float(0)
            self.api.call('cudnnSetRNNDataDescriptor',self.query_desc,0,1,n,1,self.C,seq,p.ct.byref(padding))
            work,reserve=p.S(),p.S()
            self.api.call('cudnnGetRNNTempSpaceSizes',self.q.handle,self.rnn,0,self.query_desc,p.ct.byref(work),p.ct.byref(reserve))
            need(reserve.value==0,'Prototype supports only the observed zero inference reserve')
            need(work.value<=WORKSPACE_BUDGET,'Actual cuDNN workspace exceeds fixed64MiB prototype budget')
            need(self.q.total+WORKSPACE_BUDGET+n*(self.C+2*H)*4<=CAP_BYTES,'Explicit persistent+TRT scratch+I/O cap512MiB exceeded')
            self.cache[n]=work.value
            self.report.setdefault('actual_workspace_bytes_by_N',{})[str(n)]=work.value
        return self.cache[n]

    def shape(self,n):
        started=time.perf_counter()
        self.workspace_bytes=self.query(n);p=self.p;self.sequence=(p.I*1)(n);self.padding=p.ct.c_float(0)
        for descriptor,width in ((self.xdesc,self.C),(self.ydesc,2*H)):
            self.api.call('cudnnSetRNNDataDescriptor',descriptor,0,1,n,1,width,self.sequence,p.ct.byref(self.padding))
        self.current_N=n
        self.report.setdefault('shape_descriptor_query_wall_ms',[]).append({'N':n,'wall_ms':(time.perf_counter()-started)*1000})

    def enqueue(self,n,x,y,workspace,stream):
        need(not self.closed and self.current_N==n,'on_shape_change must finish before enqueue')
        need(isinstance(stream,int)and stream>=0 and x>0 and y>0,'Invalid borrowed pointers/stream')
        need(self.workspace_bytes==0 or workspace>0,'TensorRT workspace pointer required')
        p=self.p;self.last_stream=stream
        self.api.call('cudnnSetStream',self.q.handle,p.P(stream))
        # Exactly one whole-sequence NN call; no per-token Python loop/copy/sync.
        self.api.call('cudnnRNNForward',self.q.handle,self.rnn,0,None,self.xdesc,p.P(x),self.ydesc,p.P(y),self.hdesc,None,None,self.cdesc,None,None,self.weight_bytes,self.weights,self.workspace_bytes,p.P(workspace),0,None)

    def close(self):
        if self.closed:return
        self.closed=True
        if self.q is not None:
            # Teardown only. Borrowed TRT stream is synchronized, never destroyed.
            error=None
            try:
                if self.last_stream is not None:self.api.call('cudaStreamSynchronize',self.p.P(self.last_stream))
            except BaseException as caught:error=caught
            finally:self.q.close()
            need(self.report.get('cleanup_complete'),'Private cuDNN resource cleanup failed')
            if error is not None:raise error

def make_plugin_classes(trt):
    """Explicit class creation, still no CDLL/CUDA call. Pass the existing TRT module."""
    need(re.match(r'^10\.4(?:\.|$)',trt.__version__),'This prototype requires the tested TRT10.4 old class API')
    if id(trt)in _CLASSES:return _CLASSES[id(trt)]
    import numpy as np

    def own_weights(values,meta):
        shapes={k:tuple(v) for k,v in meta['weight_shapes'].items()};result={}
        need(set(values)==set(shapes),'Exactly original W/R/B are required')
        for name,shape in shapes.items():
            v=np.array(values[name],copy=True,order='C')
            need(v.dtype==np.float32 and v.size==int(np.prod(shape))and np.isfinite(v).all(),'Original finite FP32 weights only')
            v=v.reshape(shape);need(hashlib.sha256(v.tobytes()).hexdigest()==meta['original_weight_sha256'][name],'Exact original LSTM weights required')
            v.flags.writeable=False;result[name]=v
        return result

    def static_contract(inputs,outputs,meta):
        C=meta['C'];N_LIMIT=meta['N_limit']
        need(len(inputs)==len(outputs)==1,'One X and one Y only')
        x,y=inputs[0],outputs[0];n=int(x.dims[0])
        need(tuple(x.dims)==(n,1,C)and tuple(y.dims)==(n,2,1,H)and 1<=n<=N_LIMIT,'Exact B1/fullN ABI required')
        need(x.type==y.type==trt.DataType.FLOAT and x.format==y.format==trt.TensorFormat.LINEAR,'FP32 LINEAR only')
        return n

    class Plugin(trt.IPluginV3,trt.IPluginV3OneCore,trt.IPluginV3OneBuild,trt.IPluginV3OneRuntime):
        def __init__(self,weights,meta):
            for base in (trt.IPluginV3,trt.IPluginV3OneCore,trt.IPluginV3OneBuild,trt.IPluginV3OneRuntime):base.__init__(self)
            self.plugin_name=NAME;self.plugin_version=VERSION;self.plugin_namespace=NAMESPACE;self.num_outputs=1
            self.metadata=checked_metadata(meta);self.weights=own_weights(weights,self.metadata);self._state=None;self._mutex=threading.RLock();self._destroyed=False
            self._fields=None;self._field_storage=None;self.lifecycle_report={'prototype':True,'ready':False,'NN_validation':False}
            with _LOCK:_INSTANCES[id(self)]=self

        def get_capability_interface(self,kind):
            return self if kind in (trt.PluginCapabilityType.CORE,trt.PluginCapabilityType.BUILD,trt.PluginCapabilityType.RUNTIME)else None

        def get_valid_tactics(self):return []

        def set_tactic(self,tactic):need(tactic==0,'Only the default STANDARD/FMA tactic is supported')

        def clone(self):return Plugin(self.weights,self.metadata) # CPU-owned copies only; no device resource sharing/allocation

        def attach_to_context(self,resource_context):return self.clone()

        def get_output_data_types(self,input_types):
            need(len(input_types)==1 and input_types[0]==trt.DataType.FLOAT,'FP32 only');return [trt.DataType.FLOAT]

        def get_output_shapes(self,inputs,shape_inputs,expr_builder):
            need(len(inputs)==1 and len(inputs[0])==3 and not shape_inputs,'Only X[N,1,C]')
            return [trt.DimsExprs([inputs[0][0],expr_builder.constant(2),expr_builder.constant(1),expr_builder.constant(H)])]

        def supports_format_combination(self,pos,in_out,num_inputs):
            if num_inputs!=1 or len(in_out)!=2 or pos not in (0,1):return False
            desc=in_out[pos].desc
            return desc.type==trt.DataType.FLOAT and desc.format==trt.TensorFormat.LINEAR

        def configure_plugin(self,inputs,outputs):
            C=self.metadata['C'];N_LIMIT=self.metadata['N_limit']
            need(len(inputs)==len(outputs)==1,'Only one X/Y')
            for dynamic,width,rank in [(inputs[0],C,3),(outputs[0],H,4)]:
                desc=dynamic.desc;need(desc.type==trt.DataType.FLOAT and desc.format==trt.TensorFormat.LINEAR,'FP32 LINEAR only')
                need(len(desc.dims)==rank,'Wrong rank')
                for shape in (dynamic.min,dynamic.max):
                    n=int(shape[0]);need(1<=n<=N_LIMIT,'Bounded N only')
                    need(tuple(shape)==((n,1,C)if rank==3 else(n,2,1,H)),'Fixed B/C/H and output layout required')

        def _runtime(self):
            need(not self._destroyed,'Destroyed plugin')
            if self._state is None:self._state=_Runtime(self.weights,np,self.metadata)
            return self._state

        def get_workspace_size(self,inputs,outputs):
            self.configure_plugin(inputs,outputs);return WORKSPACE_BUDGET

        def on_shape_change(self,inputs,outputs):
            n=static_contract(inputs,outputs,self.metadata)
            with self._mutex:self._runtime().shape(n)

        def enqueue(self,input_desc,output_desc,inputs,outputs,workspace,stream):
            n=static_contract(input_desc,output_desc,self.metadata);need(len(inputs)==len(outputs)==1,'One input/output pointer')
            with self._mutex:
                need(self._state is not None,'Runtime must initialize before enqueue')
                self._state.enqueue(n,int(inputs[0]),int(outputs[0]),int(workspace),int(stream))

        def get_fields_to_serialize(self):
            # Keep own immutable backing arrays alive for the returned collection.
            self._field_storage={'metadata':np.frombuffer(json.dumps(self.metadata,sort_keys=True,separators=(',',':')).encode('utf-8'),dtype=np.int8).copy(),**{name:v.reshape(-1)for name,v in self.weights.items()}}
            self._fields=trt.PluginFieldCollection([trt.PluginField(name,value,trt.PluginFieldType.CHAR if name=='metadata'else trt.PluginFieldType.FLOAT32)for name,value in self._field_storage.items()])
            return self._fields

        def destroy(self):
            with self._mutex:
                if self._destroyed:return
                self._destroyed=True
                if self._state is not None:
                    try:self._state.close()
                    except BaseException as error:
                        self.lifecycle_report['explicit_teardown_error']=type(error).__name__+': '+str(error)
                        raise
                    finally:self.lifecycle_report.update(self._state.report);self._state=None

    class Creator(trt.IPluginCreatorV3One):
        def __init__(self):
            trt.IPluginCreatorV3One.__init__(self);self.name=NAME;self.plugin_version=VERSION;self.plugin_namespace=NAMESPACE
            self._empty={'metadata':np.empty(0,dtype=np.int8),**{name:np.empty(0,dtype=np.float32)for name in ('W','R','B')}}
            self.field_names=trt.PluginFieldCollection([trt.PluginField(name,value,trt.PluginFieldType.CHAR if name=='metadata'else trt.PluginFieldType.FLOAT32)for name,value in self._empty.items()])

        def create_plugin(self,name,fc,phase):
            need(phase in (trt.TensorRTPhase.BUILD,trt.TensorRTPhase.RUNTIME),'Unknown phase')
            fields={f.name:f for f in fc};need(len(fields)==len(fc)==4 and set(fields)=={'metadata','W','R','B'},'Exact serialized fields required')
            need(fields['metadata'].type==trt.PluginFieldType.CHAR,'Metadata CHAR required')
            payload=np.array(fields['metadata'].data,copy=True).tobytes();need(len(payload)<=4096,'Bounded metadata')
            # ONNX parser includes a C-string terminator; runtime serialization may omit it.
            payload=payload.rstrip(b'\x00')
            need(payload and b'\x00' not in payload,'Embedded metadata NUL forbidden')
            meta=checked_metadata(json.loads(payload.decode('utf-8')))
            weights={}
            for key in ('W','R','B'):
                need(fields[key].type==trt.PluginFieldType.FLOAT32,'FP32 weights only');weights[key]=np.array(fields[key].data,copy=True)
            return Plugin(weights,meta) # never retain TensorRT-owned field buffers

    _CLASSES[id(trt)]=(Plugin,Creator);return Plugin,Creator

def register_creator(trt):
    """Call explicitly before building/deserializing a plan in the SAME Python process."""
    with _LOCK:
        if id(trt)in _REGISTERED:return _REGISTERED[id(trt)]
        _,Creator=make_plugin_classes(trt);creator=Creator();registry=trt.get_plugin_registry()
        need(registry.get_creator(NAME,VERSION,NAMESPACE)is None,'Refuse collision with an existing creator')
        need(registry.register_creator(creator,NAMESPACE),'Own creator registration failed')
        _REGISTERED[id(trt)]=creator;return creator

def close_all_lifecycle():
    """Explicit unit-worker teardown BEFORE closing TRT pool/context or its stream.

    Caller must finish its original inference synchronization first. Only this
    module's plugin instances are weakly tracked; no strong global ownership and
    no other plugins, libraries, contexts or streams are destroyed here.
    The unit worker must require cleanup_complete before calling its result valid.
    """
    with _LOCK:instances=list(_INSTANCES.values())
    report={'scope':NAMESPACE,'instances':[],'errors':[],'cleanup_complete':True}
    for instance in instances:
        try:instance.destroy()
        except BaseException as error:report['errors'].append({'instance':id(instance),'error':type(error).__name__+': '+str(error)})
        report['instances'].append({'instance':id(instance),'destroyed':instance._destroyed,'state_released':instance._state is None,'lifecycle':dict(instance.lifecycle_report)})
    report['cleanup_complete']=not report['errors']and all(r['destroyed']and r['state_released']and r['lifecycle'].get('cleanup_complete',True)and not r['lifecycle'].get('explicit_teardown_error')for r in report['instances'])
    return report
