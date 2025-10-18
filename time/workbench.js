document.addEventListener('DOMContentLoaded', function() {
  const { createApp, ref, computed } = Vue;
  const { ElMessage, ElMessageBox } = ElementPlus;

  createApp({
    setup() {
      // 状态管理
      const showSidebar = ref(false);
      const currentMenu = ref("workbench");
      const userInfo = ref({
        name: "张三",
        role: "普通用户",
        avatar: "https://picsum.photos/id/1005/200/200"
      });

      const projectList = ref([
        {
          id: 1,
          name: "2024年Q3销售预测项目",
          desc: "基于2023-2024年销售数据，预测Q3季度各产品销售额",
          createTime: "2024-06-15",
          datasetName: "销售数据2023-2024.csv",
          modelType: "Autoformer",
          status: "completed"
        },
        {
          id: 2,
          name: "用户增长趋势预测",
          desc: "分析APP日活用户数据，预测未来30天用户增长趋势",
          createTime: "2024-07-02",
          datasetName: "APP日活数据.xlsx",
          modelType: "Patch-TST",
          status: "completed"
        },
        {
          id: 3,
          name: "库存需求预测（测试中）",
          desc: "基于历史库存消耗数据，预测各SKU的月度库存需求",
          createTime: "2024-07-10",
          datasetName: "库存消耗数据.json",
          modelType: "LSTM",
          status: "running"
        }
      ]);

      const searchKeyword = ref("");
      const selectedStatus = ref("");
      const selectedTimeRange = ref("");
      const currentPage = ref(1);
      const pageSize = ref(6);
      const showDeleteDialog = ref(false);
      const deleteProjectId = ref(null);

      // 方法
      const goToProfile = () => {
        currentMenu.value = "profile";
        if (window.innerWidth <= 992) showSidebar.value = false;
        ElMessage.info("跳转至个人信息页面");
      };

      const toggleSidebar = () => {
        showSidebar.value = !showSidebar.value;
      };

      const switchMenu = (menuKey) => {
        currentMenu.value = menuKey;
        if (window.innerWidth <= 992) showSidebar.value = false;
        switch(menuKey) {
          case "workbench": 
            ElMessage.info("当前为工作台页面"); 
            break;
          case "data": 
            ElMessage.info("跳转至数据管理页面");
            window.location.href = "data_gover.html"; // 添加页面跳转
            break;
          case "model": 
            ElMessage.info("跳转至模型管理页面");
            window.location.href = "model_gover.html"; // 添加页面跳转
            break;
          case "profile": 
            ElMessage.info("跳转至个人信息页面"); 
            break;
        }
      };

      const handleLogout = () => {
        ElMessageBox.confirm("确定退出登录？", "确认", {
          confirmButtonText: "确定",
          cancelButtonText: "取消",
          type: "warning"
        }).then(() => ElMessage.success("已退出登录"));
      };

      const filteredProjects = computed(() => {
        let res = [...projectList.value];
        if (searchKeyword.value) {
          const kw = searchKeyword.value.toLowerCase();
          res = res.filter(item => 
            item.name.toLowerCase().includes(kw) || item.desc.toLowerCase().includes(kw)
          );
        }
        if (selectedStatus.value) {
          res = res.filter(item => item.status === selectedStatus.value);
        }
        return res;
      });

      const getStatusText = (status) => ({
        completed: "已完成",
        running: "计算中",
        failed: "计算失败"
      }[status] || "未知");

      const getTagType = (status) => ({
        completed: "success",
        running: "info",
        failed: "danger"
      }[status] || "default");

      const handleCreateProject = () => {
        ElMessage.info("跳转至新建项目页面");
        window.location.href = "new_project.html";
      };
      const handleSearch = () => currentPage.value = 1;
      const handleFilter = () => currentPage.value = 1;
      const handleViewResult = (id) => ElMessage.info(`查看项目${id}结果`);
      const handleRetrain = (id) => ElMessage.info(`重新训练项目${id}`);
      const handleSizeChange = (val) => { pageSize.value = val; currentPage.value = 1; };
      const handleCurrentChange = (val) => currentPage.value = val;
      const handleDeleteProject = (id) => { deleteProjectId.value = id; showDeleteDialog.value = true; };
      const confirmDelete = () => {
        projectList.value = projectList.value.filter(item => item.id !== deleteProjectId.value);
        showDeleteDialog.value = false;
        ElMessage.success("项目已删除");
      };

      return {
        showSidebar, currentMenu, userInfo, goToProfile, toggleSidebar, switchMenu, handleLogout,
        searchKeyword, selectedStatus, selectedTimeRange, currentPage, pageSize, showDeleteDialog,
        filteredProjects, getStatusText, getTagType, handleCreateProject, handleSearch, handleFilter,
        handleViewResult, handleRetrain, handleSizeChange, handleCurrentChange, handleDeleteProject, confirmDelete
      };
    }
  }).use(ElementPlus).mount('#app');
});