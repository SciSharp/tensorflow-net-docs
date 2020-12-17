# 帮忙写点代码

这个项目其实是一个天坑，目前还是有点缺人滴……

一起来：

* 宣传一波
* 整点用 C# 或者 F# 写的 TensorFlow 测试用例
* 帮忙填填没写完的坑
* 到[项目里](https://github.com/SciSharp/TensorFlow.NET/issues)写点 issues

### 如何搞定 Git

在你的 fork 仓库里面将 SciSharp/TensorFlow.NET 作为你的上游 upstream：

```bash
git remote add upstream git@github.com:SciSharp/TensorFlow.NET.git
```

更新一下你的 fork 仓库：

```bash
git pull upstream master
```

#### 更新 forked 仓库

```bash
# 确保当前的分支（branch）是 master
git checkout master

# 将【主仓库】（upstream/master）的所有改变都下载到【本地仓库】
# 此时并不会覆盖你本地仓库的内容
git pull upstream master

# 注意！这会【删除】你【本地仓库】的所有改变，相当于重新对【主仓库】（upstream）做一次 fork
git reset --hard upstream/master

# 注意！这会将你【 GitHub 上的仓库】强行同步为你【当前的本地仓库】
git push origin master --force
```
