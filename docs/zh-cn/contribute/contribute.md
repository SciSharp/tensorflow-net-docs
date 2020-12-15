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

```
# ensures current branch is master
git checkout master

# pulls all new commits made to upstream/master
git pull upstream master

# this will delete all your local changes to master
git reset --hard upstream/master

# take care, this will delete all your changes on your forked master
git push origin master --force
```
